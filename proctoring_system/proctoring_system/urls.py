"""
URL configuration for proctoring_system project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from proctoring import views
from proctoring.views import index
urlpatterns = [
    path('sessions/', views.ExamSessionListCreate.as_view(), name='session-list-create'),
    path('sessions/<int:pk>/', views.ExamSessionDetail.as_view(), name='session-detail'),
    path('proctoring/analyze/', views.analyze_video, name='analyze-video'),
    path('proctoring/analyze-frame/', views.analyze_frame, name='analyze-frame'),
    path('proctoring/violations/', views.ViolationList.as_view(), name='violation-list'),
    path('api/sessions', views.sessions, name='sessions'),
    path('', index, name='index'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
