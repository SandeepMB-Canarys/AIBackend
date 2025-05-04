from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import DocumentAnalysis
from .doc_checker import (
    initialize_openai,
    initialize_claude,
    detect_document_type,
    check_rules_against_document,
    get_token_usage,
    update_token_usage
)
import os
import logging
from .forms import DocumentAnalysisForm
from openai import RateLimitError, AuthenticationError

logger = logging.getLogger(__name__)

def analyze_document(request, document_id):
    # Get the document analysis instance or return 404
    document = get_object_or_404(DocumentAnalysis, id=document_id)
    
    # For API requests, return JSON response
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            # Check if analysis is already completed
            if document.analysis_completed:
                # Return stored results
                results_list = document.analysis_results if document.analysis_results else []
                total_rules = len(results_list)
                passed_rules = sum(1 for r in results_list if r['Status'] == '✅ Rule Met')
                failed_rules = total_rules - passed_rules
                compliance_rate = (passed_rules / total_rules * 100) if total_rules > 0 else 0
                
                return JsonResponse({
                    'status': 'success',
                    'doc_type': document.document_type,
                    'created_at': document.created_at,
                    'results': results_list,
                    'stats': {
                        'total_rules': total_rules,
                        'passed_rules': passed_rules,
                        'failed_rules': failed_rules,
                        'compliance_rate': round(compliance_rate, 1)
                    },
                    'usage': document.token_usage,
                    'report_url': document.report_url
                })
            
            return JsonResponse({
                'status': 'error',
                'message': 'Analysis not completed. Please try uploading the document again.'
            }, status=400)
            
        except Exception as e:
            logger.exception('Error retrieving analysis results')
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    # For regular requests, render the template
    return render(request, 'upload/analyze.html', {
        'document': document
    })


def download_document(request, document_id):
    """Download the original document"""
    try:
        document = get_object_or_404(DocumentAnalysis, id=document_id)
        file_path = os.path.join(settings.MEDIA_ROOT, str(document.document))
        response = FileResponse(open(file_path, 'rb'))
        response['Content-Disposition'] = f'attachment; filename="{os.path.basename(document.document.name)}"'
        return response
    except Exception as e:
        logger.exception('Error downloading document')
        return render(request, 'upload/analyze.html', {
            'error': f'Error downloading document: {str(e)}'
        })

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        form = DocumentAnalysisForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # Save form and get file
                document_analysis = form.save()
                api_key = form.cleaned_data['api_key']
                llm_model = form.cleaned_data['llm_model']
                ai_provider = form.cleaned_data['ai_provider']
                print("AI Provider:", ai_provider)
                docx_path = document_analysis.document.path
                
                # Check for rules file
                rules_excel_path = os.path.join(settings.BASE_DIR, 'static', 'Rules with Subrules (Static).xlsx')
                if not os.path.exists(rules_excel_path):
                    raise FileNotFoundError('Rules file not found. Please contact the administrator.')

                # Init LLM + embedding client
                try:
                    if ai_provider == 'openai':
                        print("Initializing OpenAI client and embeddings.", ai_provider)
                        client, embedding = initialize_openai(api_key=api_key, llm_model=llm_model)
                    elif ai_provider == 'claude':
                        print("Initializing Claude client and embeddings.", ai_provider)
                        client, embedding = initialize_claude(api_key=api_key, llm_model=llm_model)
                    else:
                        raise ValueError(f"Unknown AI provider: {ai_provider}")
                except AuthenticationError:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Invalid OpenAI API key. Please check your API key and try again.'
                    }, status=401)
                except Exception as e:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to initialize LLM client. Please try again.'
                    }, status=500)

                try:
                    # Detect document type using the selected model
                    doc_type = detect_document_type(docx_path, rules_excel_path, client, embedding, model=llm_model, provider=ai_provider)

                    # Run rule check with the selected model
                    result_df = check_rules_against_document(docx_path, rules_excel_path, doc_type, client, embedding, model=llm_model, provider=ai_provider)

                    # Get the filename for the success message
                    filename = os.path.basename(document_analysis.document.name)
                    
                    # Get the report URL
                    report_name = os.path.basename(result_df.report_path) if hasattr(result_df, 'report_path') else None
                    report_url = f'/media/report/{report_name}' if report_name else None

                    # Fetch token usage summary and update it
                    token_usage = get_token_usage()
                    total_input_tokens = token_usage["prompt_tokens"]
                    total_output_tokens = token_usage["completion_tokens"]
                    total_latency = token_usage["latency_seconds"]

                    # Calculate cost based on model
                    if llm_model == 'gpt-4o':
                        input_cost = (total_input_tokens / 1000) * 0.03  # $0.03 per 1K input tokens
                        output_cost = (total_output_tokens / 1000) * 0.06  # $0.06 per 1K output tokens
                    elif llm_model == 'gpt-3.5-turbo':
                        input_cost = (total_input_tokens / 1000) * 0.0015  # $0.0015 per 1K input tokens
                        output_cost = (total_output_tokens / 1000) * 0.002  # $0.002 per 1K output tokens
                    elif llm_model == 'gpt-4.1-mini':
                        input_cost = (total_input_tokens / 1000) * 0.04  # $0.04 per 1K input tokens
                        output_cost = (total_output_tokens / 1000) * 0.08  # $0.08 per 1K output tokens
                    elif llm_model == 'claude-3-opus-20240229':
                        input_cost = (total_input_tokens / 1000) * 0.015  # $0.015 per 1K input tokens
                        output_cost = (total_output_tokens / 1000) * 0.075  # $0.075 per 1K output tokens
                    elif llm_model == 'claude-3-sonnet-20240229':
                        input_cost = (total_input_tokens / 1000) * 0.003  # $0.003 per 1K input tokens
                        output_cost = (total_output_tokens / 1000) * 0.015  # $0.015 per 1K output tokens
                    elif llm_model == 'claude-3-haiku-20240307':
                        input_cost = (total_input_tokens / 1000) * 0.00025  # $0.00025 per 1K input tokens
                        output_cost = (total_output_tokens / 1000) * 0.00125  # $0.00125 per 1K output tokens
                    else:
                        return JsonResponse({
                            'status': 'error',
                            'message': 'Unsupported model selected.' }, status=400)
                    
                    total_cost = input_cost + output_cost

                    print("Report======>>>>>>>>>>>>>>>>>>>>>>>>>>>:", result_df.to_dict(orient='records'))
                    
                    # Calculate rules statistics
                    if result_df is not None:
                        results_list = result_df.to_dict(orient='records')
                        total_rules = len(results_list)
                        passed_rules = sum(1 for r in results_list if r['Status'] == '✅ Rule Met')
                        failed_rules = total_rules - passed_rules
                        compliance_rate = (passed_rules / total_rules * 100) if total_rules > 0 else 0
                    else:
                        results_list = []
                        total_rules = passed_rules = failed_rules = 0
                        compliance_rate = 0

                    # Store analysis results in the database
                    document_analysis.document_type = doc_type
                    document_analysis.analysis_results = results_list
                    document_analysis.token_usage = {
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_input_tokens + total_output_tokens,
                        'latency': round(total_latency, 2),
                        'total_cost': round(total_cost, 4)
                    }
                    document_analysis.total_cost = total_cost
                    document_analysis.report_url = report_url
                    document_analysis.analysis_completed = True
                    document_analysis.save()

                    return JsonResponse({
                        'status': 'success',
                        'message': f'File {filename} analyzed successfully!',
                        'document_id': document_analysis.id,
                        'detected_type': doc_type,
                        'file': {
                            'name': filename,
                            'size': document_analysis.document.size,
                            'url': document_analysis.document.url
                        },
                        'report_url': report_url,
                        'result_summary': results_list,
                        'rules_stats': {
                            'total_rules': total_rules,
                            'passed_rules': passed_rules,
                            'failed_rules': failed_rules,
                            'compliance_rate': round(compliance_rate, 1)
                        },
                        'usage': {
                            'input_tokens': total_input_tokens,
                            'output_tokens': total_output_tokens,
                            'total_tokens': total_input_tokens + total_output_tokens,
                            'latency': round(total_latency, 2),
                            'total_cost': round(total_cost, 4)
                        }
                    })

                except RateLimitError:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'OpenAI API quota exceeded. Please check your billing details or try again later.'
                    }, status=429)
                except Exception as e:
                    logger.exception('Error during document analysis')
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to analyze document. Please try again or contact support.'
                    }, status=500)

            except FileNotFoundError as e:
                logger.error(f'File not found: {str(e)}')
                return JsonResponse({
                    'status': 'error',
                    'message': str(e)
                }, status=404)

            except Exception as e:
                logger.exception('Error during file processing')
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to process file. Please try again.'
                }, status=500)

        return JsonResponse({
            'status': 'error',
            'message': 'Please correct the form errors.',
            'errors': form.errors
        }, status=400)

    else:
        form = DocumentAnalysisForm()
        return render(request, 'upload/upload.html', {'form': form})
