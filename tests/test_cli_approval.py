from types import SimpleNamespace
from importlib import import_module

from agent.objects import Answer, RefundProposal, PurchaseItem


cli = import_module("agent.agent")


def _install_cli_fakes(monkeypatch, confirmation):
    proposal = RefundProposal(
        username="user-001",
        items=[PurchaseItem(name="Keyboard")],
    )

    approval = {
        "request_id": "request-cli-test",
        "user": "user-001",
        "items": [{"name": "Keyboard"}],
        "quote": {
            "user": "user-001",
            "items": [
                {
                    "item": "Keyboard",
                    "price": "49.99",
                    "purchase_date": "2026-10-20",
                }
            ],
            "refund_amount": "49.99",
        },
    }

    response = {
        "structured_response": Answer(
            answer="I prepared a refund quote for your Keyboard.",
            conversation=False,
            refund_proposal=proposal,
        )
    }

    class FakeAgent:
        def invoke(self, *args, **kwargs):
            return response

    issued = []

    def fake_issue(**kwargs):
        issued.append(kwargs)
        return {
            "ticket_issued": True,
            "ticket_id": "TICKET-CLI-001",
            "existing": False,
        }

    monkeypatch.setattr(cli, "agent", FakeAgent())
    monkeypatch.setattr(
        cli,
        "_proposal_to_approval",
        lambda received_proposal, request_id: (approval, None),
    )
    monkeypatch.setattr(cli, "issue_approved_refund_ticket", fake_issue)
    monkeypatch.setattr(cli, "ToolTraceCallback", lambda: object())
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return confirmation

    monkeypatch.setattr(cli, "write_agent_trace", lambda **kwargs: None)
    monkeypatch.setattr("builtins.input", fake_input)

    return issued, approval, prompts


def test_cli_issues_ticket_only_after_explicit_y(monkeypatch, capsys):
    issued, approval, prompts = _install_cli_fakes(monkeypatch, "y")

    cli.run_conversation("I want to refund my Keyboard.")

    assert len(issued) == 1
    assert issued[0]["request_id"] == approval["request_id"]
    assert issued[0]["user"] == approval["user"]
    assert issued[0]["approved_quote"] == approval["quote"]

    output = capsys.readouterr().out
    assert any("Confirm this exact refund quote?" in p for p in prompts)
    assert "TICKET-CLI-001" in output
    assert "No money was transferred." in output


def test_cli_rejected_proposal_hides_agent_success_claim(monkeypatch, capsys):
    issued, _, prompts = _install_cli_fakes(monkeypatch, "y")

    monkeypatch.setattr(
        cli,
        "_proposal_to_approval",
        lambda received_proposal, request_id: (
            None,
            "purchase is outside the 30-day refund window",
        ),
    )

    cli.run_conversation("I want to refund my Keyboard.")

    output = capsys.readouterr().out

    assert issued == []
    assert prompts == []
    assert "outside the 30-day refund window" in output
    assert "No ticket was created." in output
    assert "I prepared a refund quote" not in output
    assert "Confirm this exact refund quote?" not in output


def test_cli_decline_never_issues_ticket(monkeypatch, capsys):
    issued, _, prompts = _install_cli_fakes(monkeypatch, "n")

    cli.run_conversation("I want to refund my Keyboard.")

    assert issued == []

    output = capsys.readouterr().out
    assert any("Confirm this exact refund quote?" in p for p in prompts)
    assert "Refund quote declined." in output
    assert "No ticket was created." in output
