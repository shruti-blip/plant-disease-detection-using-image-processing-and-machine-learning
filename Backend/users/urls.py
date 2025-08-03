
from django.urls import path
from .views import check_authentication
from django.urls import path
#from django.contrib.auth.views import LogoutView
from .views import (
    signup_page, login_page, contact_page, home2, 
    code_2, home, user_login, user_signup, check_authentication,classify_image,intro, logout_view,contact_view
)
urlpatterns = [
    path('home/', home, name='home'),
    path('intro/',intro, name='intro'),
    path('home2/', home2, name='home2'),
    path('code_2/', code_2, name='code_2'),  
    path('signUp/', signup_page, name='signup'),
    path('login/', login_page, name='login'),
    #path('contact/', contact_page, name='contact'),
    path('user_login/', user_login, name='user_login'),
    #path('logout/', LogoutView.as_view(next_page='/users/login/'), name='logout'),
    path('signup/', user_signup, name='signup_api'),
    path('check-auth/', check_authentication, name='check_auth'),
    path('classify/', classify_image, name='classify'),
    path('contact/', contact_view, name='contact'),
    path('logout/', logout_view, name='logout'),
]


