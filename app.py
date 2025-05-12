from flask import Flask
from routes.document_route import document_bp
from routes.search_route import search_bp
from routes.save_document import save_bp
from config import PORT

def create_app():
    app = Flask(__name__)
    # blueprint for API routes
    app.register_blueprint(document_bp, url_prefix="/api")
    app.register_blueprint(search_bp, url_prefix="/api")
    app.register_blueprint(save_bp, url_prefix="/api")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=PORT, debug=True)
