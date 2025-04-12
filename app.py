# backend/app.py
from flask import Flask
from routes.document_route import document_bp

def create_app():
    app = Flask(__name__)
    # Register blueprint for document routes
    app.register_blueprint(document_bp, url_prefix="/api")
    
    return app

if __name__ == "__main__":
    app = create_app()
    # In production, you would use a WSGI server instead of Flask's built-in server.
    app.run(host="0.0.0.0", port=8081, debug=True)
