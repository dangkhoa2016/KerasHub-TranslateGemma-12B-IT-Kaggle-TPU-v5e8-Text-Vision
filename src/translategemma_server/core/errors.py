class ValidationError(ValueError): pass
class ServiceUnavailableError(RuntimeError): pass
class WorkerNotReadyError(ServiceUnavailableError):
    def __init__(self, message, health=None):
        super().__init__(message); self.health = health
class QueueFullError(RuntimeError): pass
class StoreFullError(RuntimeError): pass
