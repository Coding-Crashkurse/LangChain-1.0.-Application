from app.models import ChatMessage


def _register_and_login(client):
    password = "SecurePass123"
    client.post(
        "/auth/register",
        data={
            "username": "loopster",
            "email": "loop@example.com",
            "password": password,
            "confirm": password,
            "skill_level": "beginner",
        },
        follow_redirects=True,
    )
    return password


def test_human_loop_roundtrip(client, app):
    _register_and_login(client)

    client.post("/chat/", data={"message": "Erklaere Listen"}, follow_redirects=True)
    with app.app_context():
        assistant_msg = (
            ChatMessage.query.filter_by(role="assistant", approved=None)
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        assert assistant_msg is not None
        first_id = assistant_msg.id

    client.post(
        "/chat/reject",
        data={"message_id": first_id, "feedback": "zu kompliziert"},
        follow_redirects=True,
    )

    with app.app_context():
        rejected = ChatMessage.query.get(first_id)
        assert rejected.approved is False
        pending_new = (
            ChatMessage.query.filter_by(role="assistant", approved=None)
            .order_by(ChatMessage.created_at.desc())
            .first()
        )
        assert pending_new is not None
        new_id = pending_new.id

    client.post(
        "/chat/approve",
        data={"message_id": new_id},
        follow_redirects=True,
    )

    with app.app_context():
        final_msg = ChatMessage.query.get(new_id)
        assert final_msg.approved is True
