from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden

from .forms import LoginForm, QueryForm, SignUpForm
from utils import get_current_weather 
from .models import ConversationHistory


def _build_recent_conversation_context(user, limit=5):
    history_qs = ConversationHistory.objects.filter(user=user).order_by("-created_at")[:limit]
    history_items = list(reversed(list(history_qs)))
    if not history_items:
        return ""

    return "\n\n".join(
        [
            f"User: {item.user_query}\nAssistant: {item.assistant_response}"
            for item in history_items
        ]
    )

def advisor_home(request):
    if not request.session.session_key:
        request.session.create()

    # 1. Retrieve data saved from a previous POST/Redirect
    answer = request.session.pop('answer', None) 
    weather = request.session.pop('weather', None)
    recent_history = []

    if request.user.is_authenticated:
        recent_history = ConversationHistory.objects.filter(user=request.user)[:10]
    
    # 2. Always provide a fresh, empty form for GET requests
    form = QueryForm()
    
    if request.method == 'POST':
        form = QueryForm(request.POST)
        if form.is_valid():
            # --- MOVED IMPORTS DEEPER ---
            # Now these only run when the user actually submits a query.
            # The homepage will load instantly.
            from llm.web_based_llm import query_llm
            from retrieval import SemanticRetriever
            # -----------------------------

            user_query = form.cleaned_data['query']
            
            # Retrieve Context
            retriever = SemanticRetriever()
            context_items = retriever.retrieve(user_query, return_with_sources=True)
            context_text = "\n\n".join(
                [
                    f"[Source: {item['citation']} | Title: {item['source']}]\n{item['text']}"
                    for item in context_items
                ]
            )
            
            # Get Weather
            current_weather = get_current_weather([(21.4826362, -158.0170701)], ["Oahu"])
            
            # Build Prompt
            history_context = ""
            if request.user.is_authenticated:
                history_context = _build_recent_conversation_context(request.user)

            prompt = (
                "You are an expert in Hawaiian culture and agriculture assisting farmers. "
                "Based on the literature context, current weather, and the user's query, "
                "write a culturally informed, actionable response to their query. "
                "Cite specific sources.\n\n"
                "--- Literature Context ---\n"
                f"{context_text}\n\n"
                "--- Current Weather ---\n"
                f"{current_weather}\n\n"
            )

            if history_context:
                prompt += (
                    "--- Recent Conversation History ---\n"
                    f"{history_context}\n\n"
                )

            prompt += (
                "--- User Query ---\n"
                f"{user_query}\n\n"
            )
            
            # Query LLM
            generated_answer = query_llm(prompt)
            if generated_answer is None:
                generated_answer = "No response was returned by the LLM."
            else:
                generated_answer = str(generated_answer).strip() or "No response was returned by the LLM."

            if request.user.is_authenticated:
                try:
                    ConversationHistory.objects.create(
                        user=request.user,
                        session_key=request.session.session_key,
                        user_query=user_query,
                        assistant_response=generated_answer,
                        retrieved_context=context_text,
                    )
                except Exception:
                    # Do not block answer rendering if history persistence fails.
                    pass
            
            # SAVE TO SESSION AND REDIRECT
            request.session['answer'] = generated_answer
            request.session['weather'] = current_weather
            
            return redirect('advisor_home')

    # 3. Render the page
    return render(request, 'advisor/home.html', {
        'form': form, 
        'answer': answer, 
        'weather': weather,
        'recent_history': recent_history,
    })


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('advisor_home')

    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('advisor_home')
    else:
        form = SignUpForm()

    return render(request, 'advisor/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('advisor_home')

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect('advisor_home')
    else:
        form = LoginForm(request)

    return render(request, 'advisor/login.html', {'form': form})


def logout_view(request):
    if request.method == 'POST':
        logout(request)
    return redirect('advisor_home')


@login_required
def conversation_history_view(request, user_id=None):
    target_user = request.user

    if user_id is not None:
        if not request.user.is_staff:
            return HttpResponseForbidden('You do not have permission to view this history.')
        target_user = User.objects.filter(id=user_id).first()
        if target_user is None:
            return HttpResponseForbidden('Requested user does not exist.')

    history_items = ConversationHistory.objects.filter(user=target_user)
    return render(
        request,
        'advisor/conversation_history.html',
        {
            'history_items': history_items,
            'history_owner': target_user,
        },
    )