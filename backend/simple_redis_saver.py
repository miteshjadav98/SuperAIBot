import pickle
import threading
from redis import Redis
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

class SimpleRedisSaver(InMemorySaver):
    """
    A custom saver that wraps LangGraph's InMemorySaver and persists 
    the state of each thread to a standard Redis instance using pickle.
    This bypasses the need for the RedisJSON module.
    """
    def __init__(self, redis_client: Redis):
        super().__init__()
        self.redis = redis_client
        self.lock = threading.Lock()

    def _load_thread(self, thread_id: str):
        data = self.redis.get(f"langgraph_thread:{thread_id}")
        if data:
            state = pickle.loads(data)
            self.storage[thread_id] = state['storage']
            
            # Load writes for this thread
            for k, v in state['writes'].items():
                self.writes[k] = v
                
            # Load blobs for this thread
            for k, v in state['blobs'].items():
                self.blobs[k] = v

    def _save_thread(self, thread_id: str):
        state = {
            'storage': self.storage.get(thread_id, {}),
            'writes': {k: v for k, v in self.writes.items() if k[0] == thread_id},
            'blobs': {k: v for k, v in self.blobs.items() if k[0] == thread_id}
        }
        self.redis.set(f"langgraph_thread:{thread_id}", pickle.dumps(state))

    def get_tuple(self, config: RunnableConfig):
        with self.lock:
            thread_id = config["configurable"]["thread_id"]
            self._load_thread(thread_id)
            return super().get_tuple(config)

    def put(self, config: RunnableConfig, *args, **kwargs):
        with self.lock:
            thread_id = config["configurable"]["thread_id"]
            self._load_thread(thread_id)
            res = super().put(config, *args, **kwargs)
            self._save_thread(thread_id)
            return res

    def put_writes(self, config: RunnableConfig, *args, **kwargs):
        with self.lock:
            thread_id = config["configurable"]["thread_id"]
            self._load_thread(thread_id)
            res = super().put_writes(config, *args, **kwargs)
            self._save_thread(thread_id)
            return res
            
    def delete_thread(self, thread_id: str):
        with self.lock:
            self._load_thread(thread_id)
            super().delete_thread(thread_id)
            self.redis.delete(f"langgraph_thread:{thread_id}")
            
    def list(self, config: RunnableConfig | None, *args, **kwargs):
        with self.lock:
            if config and "configurable" in config and "thread_id" in config["configurable"]:
                self._load_thread(config["configurable"]["thread_id"])
            return super().list(config, *args, **kwargs)
