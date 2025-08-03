from django.contrib import admin
from .models import ContactMessage
# Register your models here.
from django.contrib import admin
from .models import CustomUser

admin.site.register(CustomUser)
admin.site.register(ContactMessage)
