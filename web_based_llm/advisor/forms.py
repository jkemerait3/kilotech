from django import forms

class QueryForm(forms.Form):
    query = forms.CharField(
        label="Ask your question", 
        widget=forms.TextInput(attrs={'placeholder': 'Enter your question...'})
    )