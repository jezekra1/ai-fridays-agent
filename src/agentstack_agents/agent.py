import base64
import os
from typing import Annotated

from a2a.types import (
    Message,
    FilePart,
    FileWithBytes,
    TextPart,
)
from agentstack_sdk.a2a.extensions import (
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    FormRender,
    FormResponse,
    FormExtensionServer,
    FormExtensionSpec,
    PlatformApiExtensionServer,
    PlatformApiExtensionSpec,
)
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.platform import File
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentFinalAnswerEvent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModelParameters
from beeai_framework.tools import tool
from beeai_framework.tools.mcp import MCPTool
from mcp.client.streamable_http import streamablehttp_client

from src.agentstack_agents.visualize import prepare_flight_data, create_static_map, create_interactive_map

server = Server()


@server.agent()
async def flight_search_agent(
    input: Message,
    context: RunContext,
    llm_ext: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("openai:gpt-4o",)),
    ],
    form_extension: Annotated[FormExtensionServer, FormExtensionSpec(None)],
    _: Annotated[PlatformApiExtensionServer, PlatformApiExtensionSpec()],
):
    """Search flights"""
    await context.store(input)
    # Configure LLM from llm_extension
    llm = AgentStackChatModel(parameters=ChatModelParameters(stream=True))
    llm.set_context(llm_ext)

    prompt = f"Search flights for the user query: {input.parts[0].root.text}"
    # create a framework tool to send a form to the user
    static_png_bytes, interactive_html_bytes = None, None

    @tool
    async def ensure_all_data(form: FormRender) -> FormResponse | None:
        """
        Tool that ensures that all the required data is provided (flight dates, destination, origin, etc.).

        Args:
            form: A form that asks user for the missing inputs that are required
        Returns:
            All missing fields in a dictionary: {"start_date": ...}
        """
        return await form_extension.request_form(form=FormRender.model_validate(form))

    @tool
    async def visualize_flights(flights: list[list[str]]) -> None:
        """
        Tool that visualizes flights and saves them to a file. Use to visualize all flights from search results.
        Args:
            flights: A list of flights with waypoints (list of airport codes), for example,
                [
                    ["PRG", "LAS"],  # Direct flight
                    ["JFK", "LHR", "DXB", "SIN"],  # Flight with 2 layovers
                    # Add more flights here
                ]
        """
        nonlocal static_png_bytes, interactive_html_bytes
        # Define your flights with waypoints (list of airport codes)
        flights_gdf, airports_gdf = prepare_flight_data(flights)
        static_png_bytes = create_static_map(flights_gdf, airports_gdf)
        interactive_html_bytes = create_interactive_map(flights_gdf, airports_gdf)

    # Setup MCP Tool for searching flights
    client = streamablehttp_client("https://mcp.kiwi.com")
    kiwi_tools = await MCPTool.from_client(client)

    final_answer = []

    async for event, meta in RequirementAgent(
        llm=llm,
        tools=[*kiwi_tools, ensure_all_data, visualize_flights],
        requirements=[
            ConditionalRequirement(ensure_all_data, force_at_step=1),
            ConditionalRequirement(visualize_flights, force_after=kiwi_tools),
        ],
    ).run(prompt):
        match event:
            case RequirementAgentFinalAnswerEvent(delta=delta):
                final_answer.append(delta)
                yield delta

    final_message = AgentMessage(parts=[TextPart(text="".join(final_answer))])

    if static_png_bytes is not None:
        # Send PNG directly as base64 encoded string
        base64_string = base64.b64encode(static_png_bytes).decode("utf-8")
        file_part = FilePart(file=FileWithBytes(bytes=base64_string, mime_type="image/png", name="flights.png"))
        final_message.parts.append(file_part)
        yield file_part

    if interactive_html_bytes is not None:
        # Upload HTML file to the Agent Stack server and send it using the Agent Stack SDK
        file = await File.create(filename="flights.html", content=interactive_html_bytes, content_type="text/html")
        file_part = file.to_file_part()
        final_message.parts.append(file_part)
        yield file_part

    await context.store(final_message)


def run():
    try:
        server.run(
            host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)), context_store=PlatformContextStore()
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
