"""
Extensions module to avoid circular imports
"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from celery import Celery

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
celery = None  # Will be initialized in app factory

def init_celery(app):
    """Initialize Celery with Flask app context"""
    global celery
    celery = Celery(
        app.import_name,
        backend=app.config.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379'),
        broker=app.config.get('CELERY_BROKER_URL', 'redis://localhost:6379')
    )
    celery.conf.update(app.config)
    
    class ContextTask(celery.Task):
        """Make celery tasks work with Flask app context"""
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery
