from __future__ import annotations

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db
from app.models import User

from . import auth_bp
from .forms import LoginForm, RegistrationForm


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("chat.index"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data.strip(),
            email=form.email.data.lower(),
            password_hash=generate_password_hash(form.password.data),
            skill_level=form.skill_level.data,
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Registration successful. Welcome!", "success")
        return redirect(url_for("chat.index"))

    return render_template("auth/register.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("chat.index"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        if not user or not check_password_hash(user.password_hash, form.password.data):
            flash("Login failed. Please check your credentials.", "danger")
        else:
            login_user(user)
            flash("Signed in successfully.", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("chat.index"))

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Signed out.", "info")
    return redirect(url_for("auth.login"))
