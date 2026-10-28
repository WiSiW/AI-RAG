import os

os.environ["ZHIYUAN_DATABASE_URL"] = "sqlite:///./test_zhiyuan.db"
os.environ["ZHIYUAN_AUTO_CREATE_TABLES"] = "true"
os.environ["ZHIYUAN_SECRET_KEY"] = "test-secret-key-with-more-than-32-characters"
os.environ["ZHIYUAN_ADMIN_USERNAME"] = "admin"
os.environ["ZHIYUAN_ADMIN_PASSWORD"] = "test-password"
os.environ["ZHIYUAN_REDIS_URL"] = "redis://localhost:6399/15"
