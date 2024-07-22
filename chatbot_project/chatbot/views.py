
# chatbot/views.py

from django.shortcuts import render
from django.http import JsonResponse
import json
from .main import chatbot_response  # Import the function from main.py

def chatbot_view(request):
    if request.method == 'POST':
        try:
            user_message = json.loads(request.body)['message']
            bot_response = chatbot_response(user_message)  # Use the function from main.py
            return JsonResponse({'response': bot_response})
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return render(request, 'chatbot/chatbot.html')


