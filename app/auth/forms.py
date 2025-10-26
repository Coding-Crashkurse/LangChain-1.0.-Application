from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import PasswordField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError

from app.models import User

SKILL_CHOICES = [
    ("beginner", "Beginner"),
    ("mid", "Intermediate"),
    ("expert", "Expert"),
]


class RegistrationForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8)])
    confirm = PasswordField(
        "Confirm password",
        validators=[
            DataRequired(),
            EqualTo(
                "password",
                message="Passwords must match.",
            ),
        ],
    )
    skill_level = SelectField("Skill level", choices=SKILL_CHOICES, validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field: StringField) -> None:
        if User.query.filter_by(email=field.data.lower()).first():
            raise ValidationError("E-Mail ist bereits registriert.")

    def validate_username(self, field: StringField) -> None:
        if User.query.filter_by(username=field.data).first():
            raise ValidationError("Nutzername ist vergeben.")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")
