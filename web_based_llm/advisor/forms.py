from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

class QueryForm(forms.Form):
    query = forms.CharField(
        label="Ask your question", 
        widget=forms.TextInput(attrs={'placeholder': 'Enter your question...'})
    )


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=False)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


class LoginForm(AuthenticationForm):
    username = forms.CharField(max_length=150)