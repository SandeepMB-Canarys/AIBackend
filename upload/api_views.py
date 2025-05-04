from rest_framework import viewsets, status, parsers
from rest_framework.response import Response
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .models import DocumentAnalysis
from .serializers import DocumentAnalysisSerializer

class DocumentAnalysisViewSet(viewsets.ModelViewSet):
    queryset = DocumentAnalysis.objects.all()
    serializer_class = DocumentAnalysisSerializer
    http_method_names = ['get', 'post']  # Only allow GET and POST methods
    parser_classes = (parsers.MultiPartParser, parsers.FormParser)

    @swagger_auto_schema(
        operation_description="Upload a document for analysis",
        manual_parameters=[
            openapi.Parameter(
                name='document',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                required=True,
                description='Document file to analyze'
            ),
            openapi.Parameter(
                name='ai_provider',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=True,
                description='AI provider name'
            ),
            openapi.Parameter(
                name='llm_model',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=True,
                description='LLM model name'
            ),
            openapi.Parameter(
                name='api_key',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=True,
                description='API key for the AI provider'
            ),
        ],
        responses={
            201: DocumentAnalysisSerializer,
            400: "Bad Request",
            500: "Internal Server Error"
        },
        tags=['Document Analysis']
    )
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            self.perform_create(serializer)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Get analysis results for a document",
        responses={
            200: "Analysis results retrieved successfully",
            404: "Document not found"
        },
        tags=['Document Analysis']
    )
    @action(detail=True, methods=['get'])
    def analyze(self, request, pk=None):
        try:
            document = self.get_object()
            # Add your analysis logic here
            return Response({
                "status": document.status,
                "document_id": document.id,
                "message": "Document analysis completed"
            })
        except DocumentAnalysis.DoesNotExist:
            return Response(
                {"error": "Document not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
