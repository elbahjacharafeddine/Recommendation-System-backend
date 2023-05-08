from django.urls import path, include
from django.contrib import admin
from api.authentication.viewsets.social_login import GithubSocialLogin
from api.user.viewsets import ImageView, CommentaireView, EditProfileView, UserProfileView, RestPasswordView, \
    ChangePasswordView, PredictionOfUser, StatistiqueOfUser
from api.user.api_potatoes import ImageViewPotato
from api.user.viewsets import UserViewSet
urlpatterns = [
    path('admin/', admin.site.urls),
    path("api/users/", include(("api.routers", "api"), namespace="api")),
    # path('api/users/edit/<int:pk>/update_user/', UserViewSet.as_view({'put': 'update'}), name='update_user'),
    path('api/users/edit/update_user/', UserViewSet.as_view({'put': 'update'}), name='update_user'),
    path('api/users/delete/<int:pk>/', UserViewSet.as_view({'delete': 'destroy'}), name='delete_user'),
    path('api/users/getUsers/', UserViewSet.as_view(actions={'get': 'list'}), name='getUsers'),
    path("api/sessions/oauth/github/", GithubSocialLogin.as_view(), name="github_login"),
    path('api/predictionsbymonth/', UserViewSet.as_view(actions={'get': 'getPrecisions_par_mois'}), name='getPrecisions_par_mois'),
    path('api/commentarybymonth/', UserViewSet.as_view(actions={'get': 'getCommentaires_par_mois'}), name='getCommentaires_par_mois'),
    path('api/usersbymonth/', UserViewSet.as_view(actions={'get': 'getUsers_par_mois'}), name='getUsers_par_mois'),
    path('api/nbofusers/', UserViewSet.as_view(actions={'get': 'count_users'}), name='count_users'),
    path('api/nbofcommentary/', UserViewSet.as_view(actions={'get': 'count_commentaires'}), name='count_commentaires'),
    path('api/nbofpredictions/', UserViewSet.as_view(actions={'get': 'count_predictions'}), name='count_predictions'),
    path('api/nbofTodaysUsers/', UserViewSet.as_view(actions={'get': 'count_todays_users'}), name='count_todays_users'),
    path('api/nbofTodaysCommentaries/', UserViewSet.as_view(actions={'get': 'count_todays_commentaires'}), name='count_todays_commentaires'),
    path('api/nbofTodaysPredictions/', UserViewSet.as_view(actions={'get': 'count_todays_predictions'}), name='count_todays_predictions'),
    path('api/nbofPredictionsbyType/', UserViewSet.as_view(actions={'get': 'count_predictions_by_type'}), name='count_predictions_by_type'),
    path('api/comments/<int:user_id>/', UserViewSet.as_view(actions={'get': 'comments_by_user'}), name='comments_by_user'),
    path('upload', ImageView.as_view(), name="test"),
    path('test', ImageView.as_view(), name="ttest"),
    path('predict-potato', ImageViewPotato.as_view(), name="predicting"),
    path('save-commentaire', CommentaireView.as_view(),name ="save_commentaire"),
    path("edit-profile", EditProfileView.as_view(), name= "edit-profile"),
    path("user-profile", UserProfileView.as_view(), name="user-profile"),
    path('rest-password', RestPasswordView.as_view(), name="rest-password"),
    path('change-password', ChangePasswordView.as_view(), name="change-password"),
    path('prediction-user', PredictionOfUser.as_view(), name="prediction-of-user"),
    path('stati-user', StatistiqueOfUser.as_view(), name="user-statistique"),
]
