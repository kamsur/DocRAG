import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QScrollArea, QFrame, QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from groq import Groq
from dotenv import load_dotenv
import shutil

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "policies"

# Ensure data directory exists
os.makedirs(DATA_PATH, exist_ok=True)

# Load environment variables if needed
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY', 'KEY_HERE')

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# LLM setup
client = Groq(api_key=groq_api_key)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

class ChatMessage(QWidget):
    def __init__(self, sender, text, is_user):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        msg_layout = QVBoxLayout()
        msg_layout.setContentsMargins(0, 0, 0, 0)
        msg_widget = QWidget()
        msg_widget.setLayout(msg_layout)
        sender_label = QLabel(sender)
        sender_label.setStyleSheet("font-size: 12px; color: #666; font-weight: bold;")
        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        message_label.setStyleSheet(
            f"""
            padding: 10px;
            border-radius: 16px;
            color: black;
            font-size: 13px;
            background-color: {'#dcf8c6' if is_user else '#e6e6e6'};
            """
        )
        alignment = Qt.AlignRight if not is_user else Qt.AlignLeft
        msg_layout.addWidget(sender_label, alignment=alignment)
        msg_layout.addWidget(message_label, alignment=alignment)
        alignment_layout = QHBoxLayout()
        alignment_layout.setContentsMargins(0, 5, 0, 5)
        if is_user:
            alignment_layout.addWidget(msg_widget, alignment=Qt.AlignLeft)
            alignment_layout.addStretch()
        else:
            alignment_layout.addStretch()
            alignment_layout.addWidget(msg_widget, alignment=Qt.AlignRight)
        self.setLayout(alignment_layout)

class UploadWorker(QThread):
    finished = Signal(str)
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    def run(self):
        ext = os.path.splitext(self.file_path)[1].lower()
        filename = os.path.basename(self.file_path)
        dest_path = os.path.join(DATA_PATH, filename)
        shutil.copy2(self.file_path, dest_path)
        if ext == '.pdf':
            loader = UnstructuredPDFLoader(dest_path)
            raw_documents = loader.load()
        elif ext == '.txt':
            loader = TextLoader(dest_path)
            raw_documents = loader.load()
        elif ext == '.csv':
            loader = CSVLoader(dest_path)
            raw_documents = loader.load()
        else:
            self.finished.emit("Unsupported file type.")
            return
        chunks = text_splitter.split_documents(raw_documents)
        documents = []
        metadata = []
        ids = []
        for i, chunk in enumerate(chunks):
            documents.append(chunk.page_content)
            ids.append(f"ID{os.urandom(8).hex()}_{i}")
            metadata.append(chunk.metadata)
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            self.finished.emit(f"Uploaded and indexed {filename}.")
        else:
            self.finished.emit("No content found in file.")

class LLMWorker(QThread):
    finished = Signal(str)
    def __init__(self, user_query, conversation_history):
        super().__init__()
        self.user_query = user_query
        self.conversation_history = conversation_history
    def run(self):
        results = collection.query(
            query_texts=[self.user_query],
            n_results=4
        )
        system_prompt = f"""
You are a helpful assistant. You answer questions about documents that provide technical guidelines, license guidelines, etc. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
--------------------
The data:
{results['documents']}
"""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": self.user_query})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        assistant_reply = response.choices[0].message.content
        self.finished.emit(assistant_reply)

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AskDoc Chatbot")
        self.setFixedSize(500, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        chat_wrapper = QFrame()
        chat_wrapper.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
        """)
        chat_wrapper.setFrameShape(QFrame.StyledPanel)
        chat_layout = QVBoxLayout(chat_wrapper)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #fafafa; border:none;")
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
        self.scroll_area_layout.setContentsMargins(15, 15, 15, 15)
        self.scroll_area_layout.setSpacing(8)
        self.scroll_area_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_area_widget)
        chat_layout.addWidget(self.scroll_area)
        self.responding_label = QLabel("Responding...")
        self.responding_label.setAlignment(Qt.AlignCenter)
        self.responding_label.setStyleSheet("color: #007bff; font-size: 13px; padding: 2px;")
        self.responding_label.setVisible(False)
        chat_layout.addWidget(self.responding_label)
        upload_button = QPushButton("Upload Document (PDF, TXT, CSV)")
        upload_button.setCursor(Qt.PointingHandCursor)
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 8px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        upload_button.clicked.connect(self.upload_file)
        chat_layout.addWidget(upload_button)
        input_box = QFrame()
        input_box.setStyleSheet("background-color: white; border-top: 1px solid #ddd;")
        input_box.setFixedHeight(70)
        input_layout = QHBoxLayout(input_box)
        input_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 20px;
                padding: 12px;
                font-size: 14px;
                color: black;
            }
            QLineEdit:focus {
                border-color: #007bff;
            }
        """)
        self.chat_input.setMinimumHeight(40)
        self.chat_input.returnPressed.connect(self.send_message)
        send_button = QPushButton("Send")
        send_button.setCursor(Qt.PointingHandCursor)
        send_button.setMinimumHeight(40)
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 0px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        chat_layout.addWidget(input_box)
        main_layout.addWidget(chat_wrapper)
        self.llm_thread = None
        self.upload_thread = None
        self.conversation_history = []
    def append_message(self, text, sender_type):
        sender = "Me" if sender_type == 'user' else "AskDoc Bot"
        is_user = sender_type == 'user'
        message_widget = ChatMessage(sender, text, is_user)
        self.scroll_area_layout.insertWidget(self.scroll_area_layout.count()-1, message_widget)
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum()))
    def send_message(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.append_message(text, 'user')
        self.chat_input.clear()
        self.responding_label.setVisible(True)
        self.llm_thread = LLMWorker(text, self.conversation_history)
        self.llm_thread.finished.connect(self._show_response)
        self.llm_thread.start()
    def _show_response(self, content):
        self.append_message(content, 'bot')
        self.responding_label.setVisible(False)
        self.conversation_history.append({"role": "user", "content": self.chat_input.text()})
        self.conversation_history.append({"role": "assistant", "content": content})
        self.llm_thread = None
    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Document", "", "Documents (*.pdf *.txt *.csv)")
        if file_path:
            self.responding_label.setText("Uploading and indexing document...")
            self.responding_label.setVisible(True)
            self.upload_thread = UploadWorker(file_path)
            self.upload_thread.finished.connect(self._show_upload_result)
            self.upload_thread.start()
    def _show_upload_result(self, msg):
        self.append_message(msg, 'bot')
        self.responding_label.setText("Responding...")
        self.responding_label.setVisible(False)
        self.upload_thread = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec()) 