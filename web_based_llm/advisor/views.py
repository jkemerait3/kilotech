from django.shortcuts import render, redirect
from .forms import QueryForm
from retrieval import SemanticRetriever
from llm.local_llm import query_llm
from utils import get_current_weather # Keep your weather logic
import os
from django.conf import settings

# Construct the absolute path to your data
data_path = os.path.join(settings.BASE_DIR, 'data', 'hawaiian_chunks')

# Initialize with the absolute path
retriever = SemanticRetriever([data_path])

def advisor_home(request):
    # 1. Retrieve data saved from a previous POST/Redirect
    answer = request.session.pop('answer', None) 
    weather = request.session.pop('weather', None)
    
    # 2. Always provide a fresh, empty form for GET requests (clears the box on refresh)
    form = QueryForm()
    
    if request.method == 'POST':
        form = QueryForm(request.POST)
        if form.is_valid():
            user_query = form.cleaned_data['query']
            
            # Retrieve Context
            context_chunks = retriever.retrieve(user_query)
            context_text = "\n\n".join(context_chunks)
            
            # Get Weather
            current_weather = get_current_weather([(21.4826362, -158.0170701)], ["Oahu"])
            
            # Build Prompt
            prompt = (
                "You are an expert in Hawaiian culture and agriculture assisting farmers. "
                "Based on the literature context, current weather, and the user's query, "
                "write a culturally informed, actionable response to their query. "
                "Cite specific sources.\n\n"
                "--- Literature Context ---\n"
                f"{context_text}\n\n"
                "--- Current Weather ---\n"
                f"{current_weather}\n\n"
                "--- User Query ---\n"
                f"{user_query}\n\n"
            )
            
            # Query LLM
            generated_answer = query_llm(prompt)
            
            # SAVE TO SESSION AND REDIRECT (The PRG Pattern)
            request.session['answer'] = generated_answer
            request.session['weather'] = current_weather
            
            # Redirect to the same view using its URL name. 
            # *Note: Ensure 'advisor_home' matches the name= attribute in your urls.py*
            return redirect('advisor_home')

    # 3. Render the page
    return render(request, 'advisor/home.html', {
        'form': form, 
        'answer': answer, 
        'weather': weather
    })