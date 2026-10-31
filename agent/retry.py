from langchain.messages import HumanMessage


def _validate_saved_response(response):
    """Validate a structured response saved in a completed checkpoint."""
    if isinstance(response, dict):
        answer = response.get("answer")
        conversation = response.get("conversation")
    else:
        answer = getattr(response, "answer", None)
        conversation = getattr(response, "conversation", None)

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(
            "Completed checkpoint has no valid structured answer."
        )

    if not isinstance(conversation, bool):
        raise RuntimeError(
            "Completed checkpoint has an invalid conversation field."
        )

    return response


def build_agent_input(
    agent,
    config,
    *,
    is_retry: bool,
    user_text: str,
    request_id: str | None = None,
):
    """Build first-attempt input or safely recover a failed operation."""
    if not is_retry:
        message = HumanMessage(content=user_text, id=request_id)
        return {"messages": [message]}

    snapshot = agent.get_state(config)

    if snapshot is None:
        raise RuntimeError(
            "No checkpoint is available. "
            "Refusing to replay the original user message."
        )

    values = snapshot.values or {}

    if request_id is not None:
        messages = values.get("messages", [])
        matching_request = any(
            isinstance(message, HumanMessage)
            and message.id == request_id
            and message.content == user_text
            for message in messages
        )

        if not matching_request:
            raise RuntimeError(
                "Checkpoint does not match the pending request. "
                "Refusing to resume or recover a different operation."
            )

    if snapshot.next:
        return None

    saved_response = values.get("structured_response")

    if saved_response is None:
        raise RuntimeError(
            "The graph completed, but no recoverable structured response "
            "was saved. Refusing to replay the original user message."
        )

    return _validate_saved_response(saved_response)
