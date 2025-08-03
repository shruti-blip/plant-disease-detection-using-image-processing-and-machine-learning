from rest_framework import serializers
from .models import CustomUser

class SignupSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ["email", "username", "password"]

    def create(self, validated_data):
        user = CustomUser.objects.create(
            email=validated_data["email"],
            username=validated_data["username"]
        )
        user.set_password(validated_data["password"])  # Hash password
        user.save()
        return user

# Fix the missing closing bracket issue
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ["id", "email", "username"]  # Include fields as needed
