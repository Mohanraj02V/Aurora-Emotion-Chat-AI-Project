# from django.contrib.auth.models import AbstractUser
# from django.db import models
# from cryptography.fernet import Fernet
# from django.conf import settings
# import base64

# class CustomUser(AbstractUser):
#     email = models.EmailField(unique=True)
#     consent_given = models.BooleanField(default=False)
#     consent_given_at = models.DateTimeField(null=True, blank=True)
#     voice_cloning_consent = models.BooleanField(default=False)
#     memory_storage_consent = models.BooleanField(default=False)
    
#     def __str__(self):
#         return self.username

# class UserSession(models.Model):
#     user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
#     session_token = models.CharField(max_length=500)
#     created_at = models.DateTimeField(auto_now_add=True)
#     expires_at = models.DateTimeField()
#     is_active = models.BooleanField(default=True)
    
#     class Meta:
#         db_table = 'user_sessions'

# class EncryptedData(models.Model):
#     user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
#     encrypted_data = models.BinaryField()
#     data_type = models.CharField(max_length=50)  # 'audio', 'transcript', 'emotion'
#     created_at = models.DateTimeField(auto_now_add=True)
    
#     class Meta:
#         db_table = 'encrypted_data'
    
#     def encrypt_field(self, data):
#         cipher_suite = Fernet(settings.SECRET_KEY[:32].encode())
#         encrypted_data = cipher_suite.encrypt(data.encode())
#         return encrypted_data
    
#     def decrypt_field(self):
#         cipher_suite = Fernet(settings.SECRET_KEY[:32].encode())
#         decrypted_data = cipher_suite.decrypt(self.encrypted_data)
#         return decrypted_data.decode()

from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from cryptography.fernet import Fernet
from django.conf import settings
import base64

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    consent_given = models.BooleanField(default=False)
    consent_given_at = models.DateTimeField(null=True, blank=True)
    voice_cloning_consent = models.BooleanField(default=False)
    memory_storage_consent = models.BooleanField(default=False)
    
    # Add related_name to avoid clashes with built-in User model
    groups = models.ManyToManyField(
        Group,
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_name="customuser_set",  # Add this
        related_query_name="customuser",  # Add this
    )
    user_permissions = models.ManyToManyField(
        Permission,
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name="customuser_set",  # Add this
        related_query_name="customuser",  # Add this
    )
    
    def __str__(self):
        return self.username

class UserSession(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    session_token = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'user_sessions'

class EncryptedData(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    encrypted_data = models.BinaryField()
    data_type = models.CharField(max_length=50)  # 'audio', 'transcript', 'emotion'
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'encrypted_data'
    
    def encrypt_field(self, data):
        cipher_suite = Fernet(settings.SECRET_KEY[:32].encode())
        encrypted_data = cipher_suite.encrypt(data.encode())
        return encrypted_data
    
    def decrypt_field(self):
        cipher_suite = Fernet(settings.SECRET_KEY[:32].encode())
        decrypted_data = cipher_suite.decrypt(self.encrypted_data)
        return decrypted_data.decode()