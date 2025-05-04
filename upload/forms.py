from django import forms
from .models import DocumentAnalysis

class DocumentAnalysisForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['ai_provider'].initial = 'openai'
        self.fields['llm_model'].initial = 'gpt-4o'

    class Meta:
        model = DocumentAnalysis
        fields = ['ai_provider', 'llm_model', 'api_key', 'document']
        widgets = {
            'ai_provider': forms.Select(attrs={
                'class': 'form-select',
                'required': True
            }),
            'llm_model': forms.Select(attrs={
                'class': 'form-select',
                'required': True
            }),
            'api_key': forms.PasswordInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your OpenAI API key',
                'required': True
            }),
            'document': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.doc,.docx,.txt',
                'required': True,
                'style': 'display: none;'
            })
        }
        labels = {
            'ai_provider': 'AI Provider',
            'llm_model': 'Select LLM Model',
            'api_key': 'OpenAI API Key',
            'document': 'Upload Document'
        }