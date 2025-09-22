import gradio as gr
import time
import os
import shutil
import sys
from typing import Optional, Dict, Any
import json

# Add app directory to Python path
sys.path.append('./app')

# Import RAG utilities directly
from rag.rag_utilities import run_rag_tasks_step_by_step, check_task_status, query_vectorstore_direct

# Configuration
UPLOAD_DIR = "./uploaded_files"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

class RAGPipelineManager:
    def __init__(self):
        self.task_ids = {}
        self.pipeline_status = "idle"
        self.uploaded_file_path = None
        self.persist_directory = None
        
    def upload_file(self, file, openai_key: str) -> tuple:
        """Upload and save file locally"""
        if not openai_key.strip():
            return None, "❌ Please provide OpenAI API key", "", "", "", ""
            
        if file is None:
            return None, "❌ Please upload a file", "", "", "", ""
            
        # Set OpenAI API key as environment variable (for container)
        os.environ["OPENAI_API_KEY"] = openai_key
        
        try:
            # Save uploaded file
            filename = os.path.basename(file.name)
            self.uploaded_file_path = os.path.join(UPLOAD_DIR, filename)
            shutil.copy2(file.name, self.uploaded_file_path)
            
            return (
                file,
                f"✅ File '{filename}' uploaded successfully!",
                "📄 File Ready",
                "⏳ Pending",
                "⏳ Pending",
                ""  # task_ids_display
            )
        except Exception as e:
            return None, f"❌ Upload failed: {str(e)}", "", "", "", ""
    
    def start_pipeline(self, file) -> tuple:
        """Start the RAG pipeline using direct function call"""
        if not self.uploaded_file_path:
            return "❌ Please upload a file first", "❌ Error", "❌ Error", "❌ Error", ""
            
        try:
            # Copy file to app directory for container access
            app_upload_dir = "./app/uploaded_files"
            os.makedirs(app_upload_dir, exist_ok=True)
            container_file_path = os.path.join(app_upload_dir, os.path.basename(self.uploaded_file_path))
            shutil.copy2(self.uploaded_file_path, container_file_path)
            
            # Prepare parameters for direct function call
            pdf_path = f"/app/uploaded_files/{os.path.basename(self.uploaded_file_path)}"
            persist_directory = f"/app/vectorstore_{int(time.time())}"
            
            # Call the function directly instead of HTTP request
            result = run_rag_tasks_step_by_step(
                pdf_path=pdf_path,
                chunk_size=1000,
                chunk_overlap=200,
                persist_directory=persist_directory
            )
            
            if result.get("status") != "error":
                self.task_ids = result.get("task_ids", {})
                self.pipeline_status = "running"
                self.persist_directory = persist_directory  # Store for later querying
                
                return (
                    "✅ Pipeline started successfully!",
                    "✅ Completed",
                    "✅ Completed" if result.get("completed_tasks") and "transform_task" in result.get("completed_tasks", []) else "🔄 Running",
                    "🔄 Running" if result.get("running_tasks") and "vectorstore_task" in result.get("running_tasks", []) else "⏳ Pending",
                    json.dumps(self.task_ids, indent=2)
                )
            else:
                return f"❌ Pipeline failed: {result.get('message')}", "❌ Error", "❌ Error", "❌ Error", ""
                
        except Exception as e:
            return f"❌ Pipeline error: {str(e)}", "❌ Error", "❌ Error", "❌ Error", ""
    
    def check_task_status_with_progress(self, task_id: str) -> tuple:
        """Check individual task status and return progress percentage"""
        try:
            result = check_task_status(task_id)
            status = result.get("status", "Error")
            
            if status == "Completed":
                return "✅ Completed", 100  # 100% progress
            elif status in ["Pending", "Running"]:
                return "🔄 Running", 50  # 50% progress (running)
            elif status == "Error":
                error_msg = result.get("error", "Unknown error")
                return f"❌ Error: {error_msg}", 0  # 0% progress (error)
            else:
                return f"🔄 {status}", 50  # Other states as running
        except Exception as e:
            return f"❌ Error: {str(e)}", 0
    
    def check_task_status(self, task_id: str) -> str:
        """Check individual task status using direct function call"""
        status, _ = self.check_task_status_with_progress(task_id)
        return status
    
    def update_progress(self) -> tuple:
        """Update progress of all tasks with progress bars"""
        if not self.task_ids:
            return (
                0, "⏳ Pending",  # PDF progress and status
                0, "⏳ Pending",  # Transform progress and status
                0, "⏳ Pending",  # VectorStore progress and status
                False  # Query enabled
            )
            
        pdf_status, pdf_progress = self.check_task_status_with_progress(self.task_ids.get("pdf_task_id", ""))
        transform_status, transform_progress = self.check_task_status_with_progress(self.task_ids.get("transform_task_id", ""))
        vectorstore_status, vectorstore_progress = self.check_task_status_with_progress(self.task_ids.get("vectorstore_task_id", ""))
        
        # Enable query if all tasks completed
        all_completed = all(status == "✅ Completed" for status in [pdf_status, transform_status, vectorstore_status])
        
        if all_completed:
            self.pipeline_status = "completed"
            
        return (
            pdf_progress, pdf_status,
            transform_progress, transform_status, 
            vectorstore_progress, vectorstore_status,
            all_completed
        )
    
    def query_rag(self, question: str, persist_dir: str) -> str:
        """Query the RAG system using direct function call"""
        if not question.strip():
            return "Please enter a question."
            
        try:
            # Use provided persist_dir or fall back to the one from pipeline
            directory = persist_dir.strip() if persist_dir.strip() else self.persist_directory
            if not directory:
                directory = "/app/vectorstore_default"
            
            # Call the function directly instead of HTTP request
            result = query_vectorstore_direct(
                question=question,
                persist_directory=directory,
                top_k=3
            )
            
            if result.get("status") == "success":
                answer = result.get("answer", "No answer found")
                sources = result.get("sources", [])
                
                formatted_response = f"**Answer:** {answer}\n\n"
                if sources:
                    formatted_response += f"**Sources:** {', '.join(sources)}"
                
                return formatted_response
            else:
                return f"Query failed: {result.get('message', 'Unknown error')}"
                
        except Exception as e:
            return f"Query error: {str(e)}"

# Initialize manager
pipeline_manager = RAGPipelineManager()

# Gradio Interface
def create_gradio_app():
    # Custom CSS for enhanced styling
    custom_css = """
    <style>
    .progress-bar .wrap {
        background: linear-gradient(90deg, #f0f0f0 0%, #e0e0e0 100%) !important;
        border-radius: 15px !important;
        height: 12px !important;
    }
    
    .progress-bar .wrap .bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 15px !important;
        transition: width 0.5s ease-in-out !important;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    .compact-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 15px;
        margin: 5px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-completed {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
    }
    
    .status-running {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
    }
    
    .status-pending {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        color: white;
    }
    
    .status-error {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }
    
    .progress-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e1e5e9;
    }
    </style>
    """
    
    with gr.Blocks(title="RAG Pipeline Manager", theme=gr.themes.Soft(), css=custom_css) as app:
        gr.HTML("<h1 style='text-align: center;'>🤖 RAG Pipeline Manager</h1>")
        gr.HTML("<p style='text-align: center;'>Upload documents, process them through RAG pipeline, and query with AI</p>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Step 1: Upload & Configure</h3>")
                
                openai_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    info="Required for embeddings and LLM"
                )
                
                file_upload = gr.File(
                    label="Upload PDF Document",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                upload_btn = gr.Button("📤 Upload File", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
            with gr.Column(scale=1):
                gr.HTML("<h3>⚙️ Step 2: Pipeline Progress</h3>")
                
                start_pipeline_btn = gr.Button("🚀 Start RAG Pipeline", variant="secondary")
                pipeline_status = gr.Textbox(label="Pipeline Status", interactive=False)
                
                # Compact Progress Dashboard
                with gr.Column():
                    # Auto-refresh status indicator
                    refresh_status_display = gr.HTML(
                        value="<div style='text-align: center; padding: 8px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 15px; font-weight: bold;'>🔄 Auto-refreshing...</div>"
                    )
                    
                    # Progress Dashboard Card
                    gr.HTML("""
                    <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                                border-radius: 15px; padding: 20px; margin: 10px 0; 
                                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                                border: 1px solid rgba(255,255,255,0.2);'>
                        <h4 style='text-align: center; margin: 0 0 15px 0; color: #2c3e50; font-size: 18px;
                                   text-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                            📊 RAG Pipeline Progress Dashboard
                        </h4>
                    </div>
                    """)
                    
                    # Compact progress layout in a responsive grid
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=150):
                            # PDF Processing Card
                            with gr.Column(elem_classes=["progress-container"]):
                                pdf_progress_label = gr.HTML(
                                    "<div style='text-align: center; font-weight: bold; color: #e74c3c; margin-bottom: 8px; font-size: 14px;'>📄 PDF Processing</div>"
                                )
                                pdf_progress_bar = gr.Slider(
                                    minimum=0, maximum=100, value=0, 
                                    show_label=False, interactive=False,
                                    elem_classes=["progress-bar"]
                                )
                                pdf_progress_text = gr.HTML(
                                    "<div style='text-align: center; margin-top: 8px;'><span class='status-badge status-pending'>⏳ Pending</span></div>"
                                )
                        
                        with gr.Column(scale=1, min_width=150):
                            # Document Transformation Card
                            with gr.Column(elem_classes=["progress-container"]):
                                transform_progress_label = gr.HTML(
                                    "<div style='text-align: center; font-weight: bold; color: #f39c12; margin-bottom: 8px; font-size: 14px;'>✂️ Text Processing</div>"
                                )
                                transform_progress_bar = gr.Slider(
                                    minimum=0, maximum=100, value=0,
                                    show_label=False, interactive=False,
                                    elem_classes=["progress-bar"]
                                )
                                transform_progress_text = gr.HTML(
                                    "<div style='text-align: center; margin-top: 8px;'><span class='status-badge status-pending'>⏳ Pending</span></div>"
                                )
                        
                        with gr.Column(scale=1, min_width=150):
                            # Vector Store Creation Card
                            with gr.Column(elem_classes=["progress-container"]):
                                vectorstore_progress_label = gr.HTML(
                                    "<div style='text-align: center; font-weight: bold; color: #27ae60; margin-bottom: 8px; font-size: 14px;'>🔍 Vector Store</div>"
                                )
                                vectorstore_progress_bar = gr.Slider(
                                    minimum=0, maximum=100, value=0,
                                    show_label=False, interactive=False,
                                    elem_classes=["progress-bar"]
                                )
                                vectorstore_progress_text = gr.HTML(
                                    "<div style='text-align: center; margin-top: 8px;'><span class='status-badge status-pending'>⏳ Pending</span></div>"
                                )
                    
                    # Overall progress summary
                    overall_progress_summary = gr.HTML(
                        """<div style='text-align: center; margin: 15px 0; padding: 10px; 
                           background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%); 
                           border-radius: 10px; font-weight: bold; color: #2c3e50;'>
                           📈 Overall Progress: 0% Complete
                        </div>"""
                    )
                
                # Collapsible Task IDs section
                with gr.Accordion("🔍 Task Details", open=False):
                    task_ids_display = gr.Code(label="Task IDs", language="json", interactive=False)
                
                refresh_btn = gr.Button("🔄 Refresh Progress", variant="tertiary")
        
        gr.HTML("<hr>")
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>💬 Step 3: Query Your Document</h3>")
                
                question_input = gr.Textbox(
                    label="Ask a question about your document",
                    placeholder="What is the main topic of this document?",
                    lines=2,
                    interactive=False  # Initially disabled
                )
                
                persist_dir_input = gr.Textbox(
                    label="Vector Store Directory (optional)",
                    placeholder="Leave empty to use default",
                    interactive=False  # Initially disabled
                )
                
                query_btn = gr.Button("🔍 Query", variant="primary", interactive=False)
                
                answer_output = gr.Markdown(
                    label="Answer",
                    value="Complete the pipeline first to enable querying."
                )
        
        # Event handlers
        def handle_upload(file, key):
            return pipeline_manager.upload_file(file, key)
        
        def handle_start_pipeline(file):
            return pipeline_manager.start_pipeline(file)
        
        def handle_refresh():
            pdf_progress, pdf_stat, transform_progress, transform_stat, vector_progress, vector_stat, completed = pipeline_manager.update_progress()
            
            # Helper function to create styled progress labels
            def create_progress_label(title, emoji, progress, color):
                return f"""<div style='text-align: center; font-weight: bold; color: {color}; margin-bottom: 8px; font-size: 14px;'>
                          {emoji} {title}: {progress}%</div>"""
            
            # Helper function to create styled status badges
            def create_status_badge(status, progress):
                if "✅ Completed" in status:
                    badge_class = "status-completed"
                    icon = "✅"
                    text = "Completed"
                elif "🔄 Running" in status:
                    badge_class = "status-running"
                    icon = "🔄"
                    text = "Running"
                elif "❌ Error" in status:
                    badge_class = "status-error"
                    icon = "❌"
                    text = "Error"
                else:
                    badge_class = "status-pending"
                    icon = "⏳"
                    text = "Pending"
                
                return f"""<div style='text-align: center; margin-top: 8px;'>
                          <span class='status-badge {badge_class}'>{icon} {text}</span></div>"""
            
            # Calculate overall progress
            overall_progress = round((pdf_progress + transform_progress + vector_progress) / 3)
            
            # Create progress labels with percentages and colors
            pdf_label = create_progress_label("PDF Processing", "📄", pdf_progress, "#e74c3c")
            transform_label = create_progress_label("Text Processing", "✂️", transform_progress, "#f39c12")
            vector_label = create_progress_label("Vector Store", "🔍", vector_progress, "#27ae60")
            
            # Create status badges
            pdf_status_html = create_status_badge(pdf_stat, pdf_progress)
            transform_status_html = create_status_badge(transform_stat, transform_progress)
            vector_status_html = create_status_badge(vector_stat, vector_progress)
            
            # Create overall progress summary
            if completed:
                overall_color = "#27ae60"
                overall_bg = "linear-gradient(135deg, #d5f4e6 0%, #a8e6cf 100%)"
                overall_icon = "✅"
            elif overall_progress > 0:
                overall_color = "#f39c12"
                overall_bg = "linear-gradient(135deg, #fef5d3 0%, #f9e79f 100%)"
                overall_icon = "🔄"
            else:
                overall_color = "#7f8c8d"
                overall_bg = "linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%)"
                overall_icon = "📈"
            
            overall_summary = f"""<div style='text-align: center; margin: 15px 0; padding: 12px; 
                                 background: {overall_bg}; border-radius: 10px; 
                                 font-weight: bold; color: {overall_color}; font-size: 16px;
                                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                 {overall_icon} Overall Progress: {overall_progress}% Complete
                              </div>"""
            
            # Update auto-refresh status based on completion
            if completed:
                refresh_status = """<div style='text-align: center; padding: 8px; 
                                   background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                                   color: white; border-radius: 10px; margin-bottom: 15px; font-weight: bold;'>
                                   ✅ Pipeline completed - Auto-refresh stopped</div>"""
                timer_active = False
            else:
                refresh_status = """<div style='text-align: center; padding: 8px; 
                                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   color: white; border-radius: 10px; margin-bottom: 15px; font-weight: bold;'>
                                   🔄 Auto-refreshing...</div>"""
                timer_active = True
            
            # Update query interface based on completion
            if completed:
                return (
                    pdf_label, pdf_progress, pdf_status_html,
                    transform_label, transform_progress, transform_status_html,
                    vector_label, vector_progress, vector_status_html,
                    overall_summary,
                    gr.update(interactive=True, placeholder="Ask a question about your document..."),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    "✅ Pipeline completed! You can now query your document.",
                    refresh_status,
                    gr.update(active=timer_active)  # Stop the timer
                )
            else:
                return (
                    pdf_label, pdf_progress, pdf_status_html,
                    transform_label, transform_progress, transform_status_html,
                    vector_label, vector_progress, vector_status_html,
                    overall_summary,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "Pipeline still running...",
                    refresh_status,
                    gr.update(active=timer_active)  # Keep timer active
                )
        
        def handle_query(question, persist_dir):
            return pipeline_manager.query_rag(question, persist_dir)
        
        # Wire up events
        upload_btn.click(
            handle_upload,
            inputs=[file_upload, openai_key],
            outputs=[file_upload, upload_status, pdf_progress_text, transform_progress_text, vectorstore_progress_text, task_ids_display]
        )
        
        start_pipeline_btn.click(
            handle_start_pipeline,
            inputs=[file_upload],
            outputs=[pipeline_status, pdf_progress_text, transform_progress_text, vectorstore_progress_text, task_ids_display]
        )
        
        # Set up auto-refresh timer (every 3 seconds) - will be controlled dynamically
        refresh_timer = gr.Timer(value=3, active=True)
        
        refresh_btn.click(
            handle_refresh,
            outputs=[
                pdf_progress_label, pdf_progress_bar, pdf_progress_text,
                transform_progress_label, transform_progress_bar, transform_progress_text,
                vectorstore_progress_label, vectorstore_progress_bar, vectorstore_progress_text,
                overall_progress_summary,
                question_input, persist_dir_input, query_btn, answer_output,
                refresh_status_display, refresh_timer
            ]
        )
        
        query_btn.click(
            handle_query,
            inputs=[question_input, persist_dir_input],
            outputs=[answer_output]
        )
        
        refresh_timer.tick(
            handle_refresh,
            outputs=[
                pdf_progress_label, pdf_progress_bar, pdf_progress_text,
                transform_progress_label, transform_progress_bar, transform_progress_text,
                vectorstore_progress_label, vectorstore_progress_bar, vectorstore_progress_text,
                overall_progress_summary,
                question_input, persist_dir_input, query_btn, answer_output,
                refresh_status_display, refresh_timer
            ]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="127.0.0.1",  # Use 127.0.0.1 instead of 0.0.0.0 to avoid network issues
        server_port=7860,
        share=True,  # Enable sharing to bypass localhost access issues
        debug=False  # Disable debug mode to avoid schema processing issues
    )
