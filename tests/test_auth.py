from app.models import User


def register_user(client, email="tester@example.com", password="SecurePass123", skill="mid"):
    return client.post(
        "/auth/register",
        data={
            "username": "tester",
            "email": email,
            "password": password,
            "confirm": password,
            "skill_level": skill,
        },
        follow_redirects=True,
    )


def test_registration_and_login_flow(client, app):
    response = register_user(client, skill="expert")
    assert response.status_code == 200
    with app.app_context():
        user = User.query.filter_by(email="tester@example.com").first()
        assert user is not None
        assert user.skill_level == "expert"

    client.get("/auth/logout", follow_redirects=True)
    login_response = client.post(
        "/auth/login",
        data={"email": "tester@example.com", "password": "SecurePass123"},
        follow_redirects=True,
    )
    assert login_response.status_code == 200
    assert b"Python explainer chat" in login_response.data
