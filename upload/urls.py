from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .api_views import DocumentAnalysisViewSet

router = DefaultRouter()
router.register(r'documents', DocumentAnalysisViewSet, basename='document')

urlpatterns = [
    path('', views.upload_file , name='upload_document'),
    path('analyze/<int:document_id>/', views.analyze_document, name='analyze_document'),
    path('download/<int:document_id>/', views.download_document, name='download_document'),
    path('api/', include(router.urls)),
]
