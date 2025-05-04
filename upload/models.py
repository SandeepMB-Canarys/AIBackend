from django.db import models

class DocumentAnalysis(models.Model):
    AI_PROVIDER_CHOICES = [
        ('openai', 'OpenAI'),
        ('claude', 'Claude AI'),
    ]
    
    LLM_MODEL_CHOICES = [
        ('gpt-4o', 'GPT-4o'),
        ('gpt-3.5-turbo', 'GPT-3.5 Turbo'),
        ('gpt-4.1-mini', 'GPT-4.1 Mini'),
        ('claude-3-opus-20240229', 'Claude 3 Opus'),
        ('claude-3-sonnet-20240229', 'Claude 3 Sonnet'),
        ('claude-3-haiku-20240307', 'Claude 3 Haiku'),
    ]
    
    ai_provider = models.CharField(max_length=20, choices=AI_PROVIDER_CHOICES, default='openai')
    llm_model = models.CharField(max_length=50, choices=LLM_MODEL_CHOICES, default='gpt-4o')
    api_key = models.CharField(max_length=500)
    document = models.FileField(upload_to='documents/', help_text='Supported formats: PDF, DOC, DOCX, TXT')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Analysis results
    document_type = models.CharField(max_length=100, null=True, blank=True)
    analysis_results = models.JSONField(null=True, blank=True)
    token_usage = models.JSONField(null=True, blank=True)
    total_cost = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    report_url = models.CharField(max_length=255, null=True, blank=True)
    analysis_completed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Document {self.id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

