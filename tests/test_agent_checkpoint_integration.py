from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
sys.path.insert(0, str(AGENT_DIR))

from objects import Answer


class StructuredAnswerFakeModel(BaseChatModel):
    """Test-only model that returns a deterministic structured tool call."""

    @property
    def _llm_type(self) -> str:
        return "structured-answer-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError(
            "The test must use structured tool binding, not plain generation."
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if len(tools) != 1:
            raise AssertionError(
                "This test model must receive only the Answer schema."
            )

        schema = tools[0]
        schema_name = getattr(
            schema,
            "name",
            getattr(schema, "__name__", None),
        )

        if schema_name != "Answer":
            raise AssertionError(
                f"Unexpected structured-output schema: {schema_name}"
            )

        def return_structured_answer(_input):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Answer",
                        "args": {
                            "answer": "Checkpoint integration succeeded.",
                            "conversation": False,
                        },
                        "id": "answer-test-call-1",
                        "type": "tool_call",
                    }
                ],
            )

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(return_structured_answer)


def test_create_agent_persists_answer_in_completed_checkpoint():
    checkpointer = InMemorySaver()

    graph = create_agent(
        model=StructuredAnswerFakeModel(),
        tools=[],
        response_format=Answer,
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": "agent-checkpoint-integration-test",
        }
    }

    response = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Run the checkpoint integration test.")
            ]
        },
        config=config,
    )

    returned_answer = response.get("structured_response")

    assert isinstance(returned_answer, Answer)
    assert returned_answer.answer == "Checkpoint integration succeeded."
    assert returned_answer.conversation is False

    snapshot = graph.get_state(config)

    assert snapshot.next == ()

    saved_answer = (snapshot.values or {}).get("structured_response")

    assert isinstance(saved_answer, Answer)
    assert saved_answer == returned_answer
