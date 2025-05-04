from rest_framework import serializers
from .models import DocumentAnalysis

class DocumentAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentAnalysis
        fields = [
            'id',
            'ai_provider',
            'llm_model',
            'api_key',
            'document',
            'created_at',
            'document_type',
            'analysis_results',
            'token_usage',
            'total_cost',
            'report_url',
            'analysis_completed',
        ]
        read_only_fields = [
            'id',
            'created_at',
            'document_type',
            'analysis_results',
            'token_usage',
            'total_cost',
            'report_url',
            'analysis_completed',
        ]
