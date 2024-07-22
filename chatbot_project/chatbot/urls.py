
# chatbot_project/urls.py

from django.contrib import admin
from django.urls import path
from chatbot.views import chatbot_view  # Import the view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('chat/', chatbot_view, name='chatbot'),  # Add this line
]

