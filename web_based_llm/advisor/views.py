from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden
from pathlib import Path

from .forms import LoginForm, QueryForm, SignUpForm
from utils import get_current_weather 
from .models import ConversationHistory
import pandas as pd


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


def _build_latest_sensor_snapshot(csv_path='eclipse_example_data.csv'):
    # Keep only prompt-relevant columns and attach units for readability in the LLM context.
    field_specs = [
        ('deviceId', 'Device ID', 'string'),
        ('sensortimestamp', 'Sensor timestamp', 'UTC ISO-8601'),
        ('temperature', 'Internal temperature', 'deg C'),
        ('humidity', 'Internal relative humidity', '% RH'),
        ('externaltemperature', 'External temperature', 'deg C'),
        ('externalhumidity', 'External relative humidity', '% RH'),
        ('pressure', 'Barometric pressure', 'Pa'),
        ('pm1', 'PM1 mass concentration', 'ug/m^3'),
        ('pm25', 'PM2.5 mass concentration', 'ug/m^3'),
        ('pm10', 'PM10 mass concentration', 'ug/m^3'),
        ('Signal', 'Cell signal strength', 'dBm'),
        ('VBat', 'Battery voltage', 'V'),
        ('pctbat', 'Battery level', '%'),
        ('gpslat', 'GPS latitude', 'decimal degrees'),
        ('gpslong', 'GPS longitude', 'decimal degrees'),
    ]

    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path

    try:
        sensor_df = pd.read_csv(csv_path)
    except Exception:
        return 'No sensor data available.'

    if sensor_df.empty:
        return 'No sensor data available.'

    timestamp_col = 'sensortimestamp' if 'sensortimestamp' in sensor_df.columns else 'Timestamp'

    if timestamp_col in sensor_df.columns:
        parsed_ts = pd.to_datetime(sensor_df[timestamp_col], errors='coerce', utc=True)
        if parsed_ts.notna().any():
            latest_row = sensor_df.loc[parsed_ts.idxmax()]
        else:
            latest_row = sensor_df.iloc[-1]
    else:
        latest_row = sensor_df.iloc[-1]

    lines = []
    for column, label, unit in field_specs:
        if column not in latest_row.index:
            continue

        value = latest_row[column]
        if pd.isna(value) or value == '':
            continue

        if isinstance(value, float):
            value_str = f'{value:.4f}'.rstrip('0').rstrip('.')
        else:
            value_str = str(value)

        lines.append(f'- {label} ({unit}): {value_str}')

    if not lines:
        return 'No sensor data available.'

    return 'Latest Eclipse sensor reading:\n' + '\n'.join(lines)

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

            ## Get latest sensor data snapshot
            sensor_data = _build_latest_sensor_snapshot('eclipse_example_data.csv')
            
            # Get Weather
            current_weather = get_current_weather([(21.4826362, -158.0170701)], ["Oahu"])
            
            # Build Prompt
            history_context = ""
            if request.user.is_authenticated:
                history_context = _build_recent_conversation_context(request.user)

            prompt = (
                "You are an expert in Hawaiian culture and agriculture assisting farmers. "
                "Based on the literature context, current weather, local sensor data, and the user's query, "
                "write a culturally informed, actionable response to their query. "
                "Include citations for all specific sources referenced at the end of your response.\n\n"
                "--- Literature Context ---\n"
                f"{context_text}\n\n"
                "--- Current Weather ---\n"
                f"{current_weather}\n\n"
                "--- Sensor Data ---\n"
                f"{sensor_data}\n\n"
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