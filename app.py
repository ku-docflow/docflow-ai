from flask import Flask
from routes.document_route import document_bp
from routes.search_route import search_bp
from config import PORT

def create_app():
    app = Flask(__name__)
    # Register blueprint for document routes
    app.register_blueprint(document_bp, url_prefix="/api")
    app.register_blueprint(search_bp, url_prefix="/api")    
    return app

if __name__ == "__main__":
    app = create_app()
    # In production, you would use a WSGI server instead of Flask's built-in server.
    app.run(host="0.0.0.0", port=PORT, debug=True)
