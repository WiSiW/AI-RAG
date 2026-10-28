from app.core.security import decrypt_secret, encrypt_secret


def test_secret_roundtrip():
    encrypted = encrypt_secret("sk-test-secret")
    assert encrypted
    assert encrypted != "sk-test-secret"
    assert decrypt_secret(encrypted) == "sk-test-secret"


def test_empty_secret():
    assert encrypt_secret(None) is None
    assert decrypt_secret(None) == ""
