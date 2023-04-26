from api.user.serializers import UserSerializer
from api.user.models import User
from rest_framework import viewsets, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError
from rest_framework import mixins
import numpy as np
import cv2
import os
from rest_framework.views import APIView
from django.conf import settings
import tensorflow as tf
from django.http import JsonResponse
from io import BytesIO
from rest_framework.decorators import action
from PIL import Image
# from rest_framework.decorators import action
from api.authentication.models import ActiveSession
from django.utils import timezone
from datetime import timedelta
from api.user.models import Prediction, Commentaire
from datetime import date
from rest_framework.permissions import AllowAny
from datetime import datetime
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.db.models.functions import ExtractMonth
from django.db.models import Count


class UserViewSet(
    viewsets.GenericViewSet, mixins.CreateModelMixin, mixins.UpdateModelMixin, mixins.DestroyModelMixin
):
    serializer_class = UserSerializer
    permission_classes = (AllowAny,)
    queryset = User.objects.all()

    error_message = {"success": False, "msg": "Error updating user"}

    # allow PUT method for update
    @action(detail=True, methods=['put'], permission_classes=[])
    def update_user(self, request, pk=None):
        user = self.get_object()
        serializer = self.get_serializer(user, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop("partial", True)
        instance = User.objects.get(id=request.data.get("userID"))
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if getattr(instance, "_prefetched_objects_cache", None):
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        user_id = request.data.get("userID")

        if not user_id:
            raise ValidationError(self.error_message)

        if self.request.user.pk != int(user_id) and not self.request.user.is_superuser:
            raise ValidationError(self.error_message)

        self.update(request)

        return Response({"success": True}, status.HTTP_200_OK)

    def destroy(self, request, *args, **kwargs):
        userID = kwargs.get('pk', None)
        if not userID:
            return Response({"error": "User ID not provided"}, status=status.HTTP_400_BAD_REQUEST)

        instance = self.queryset.filter(id=userID).first()
        if not instance:
            return Response({"error": "Object not found"}, status=status.HTTP_404_NOT_FOUND)

        self.perform_destroy(instance)
        return Response({"success": True}, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], permission_classes=[AllowAny])
    def getUsers(self, request):
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    # def getPrecisions_par_mois(self, request):
    #     predictions = Prediction.objects.all()
    #     precisions_par_mois = predictions.annotate(mois=ExtractMonth('date')).values('mois').annotate(precision_par_mois_count=Count('id')).order_by('mois')
    #     data = {'precisions_par_mois': list(precisions_par_mois)}
    #     return JsonResponse(data)

    def getPrecisions_par_mois(self, request):
        predictions = Prediction.objects.all()
        precisions_par_mois = predictions.annotate(mois=ExtractMonth('date')).values('mois').annotate(
            precision_par_mois_count=Count('id')).order_by('mois')

        precisions_list = [0] * 12
        for precision in precisions_par_mois:
            mois = precision['mois']
            count = precision['precision_par_mois_count']
            precisions_list[mois - 1] = count

        data = {'precisions_par_mois': precisions_list}
        return JsonResponse(data)

    def getCommentaires_par_mois(self, request):
        commentaires = Commentaire.objects.all()
        commentaires_par_mois = commentaires.annotate(mois=ExtractMonth('date')).values('mois').annotate(
            commentaire_par_mois_count=Count('id')).order_by('mois')

        commentaires_list = [0] * 12
        for commentaire in commentaires_par_mois:
            mois = commentaire['mois']
            count = commentaire['commentaire_par_mois_count']
            commentaires_list[mois - 1] = count

        data = {'commentaires_par_mois': commentaires_list}
        return JsonResponse(data)

    def getUsers_par_mois(self, request):
        users = User.objects.all()
        users_par_mois = users.annotate(mois=ExtractMonth('date')).values('mois').annotate(
            user_par_mois_count=Count('id')).order_by('mois')

        users_list = [0] * 12
        for user in users_par_mois:
            mois = user['mois']
            count = user['user_par_mois_count']
            users_list[mois - 1] = count

        data = {'users_par_mois': users_list}
        return JsonResponse(data)

    def count_users(self, request):
        user_count = User.objects.count()
        data = {'number_of_users': user_count}
        return JsonResponse(data)

    def count_commentaires(self, request):
        commentaire_count = Commentaire.objects.count()
        data = {'number_of_commentaires': commentaire_count}
        return JsonResponse(data)

    def count_predictions(self, request):
        prediction_count = Prediction.objects.count()
        data = {'number_of_predictions': prediction_count}
        return JsonResponse(data)

    def count_todays_users(self, count):
        today = timezone.now().date()
        start_date = datetime.combine(today, datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
        users_count = User.objects.filter(date__range=(start_date, end_date)).count()
        data = {'number_of_todays_users': users_count}
        return JsonResponse(data)

    def count_todays_commentaires(self, count):
        today = timezone.now().date()
        start_date = datetime.combine(today, datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
        commentaires_count = Commentaire.objects.filter(date__range=(start_date, end_date)).count()
        data = {'number_of_todays_commentaires': commentaires_count}
        return JsonResponse(data)

    def count_todays_predictions(self, count):
        today = timezone.now().date()
        start_date = datetime.combine(today, datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
        predictions_count = Prediction.objects.filter(date__range=(start_date, end_date)).count()
        data = {'number_of_todays_predictions': predictions_count}
        return JsonResponse(data)

    def count_predictions_by_type(self, count):
        types_list = ['Tomato___Late_blight', 'Tomato___healthy', 'Potato___healthy', 'Tomato___Early_blight',
                      'Tomato___Septoria_leaf_spot', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                      'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                      'Tomato___Target_Spot', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Apple___healthy',
                      'Potato___Early_blight', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite']
        types_count = [{'type': t, 'count': Prediction.objects.filter(classe=t).count()} for t in types_list]
        result = [
            {"type": item['type'], "count": item['count']} if item['count'] > 0 else {"type": item['type'], "count": 0}
            for item in types_count]
        return JsonResponse(result, safe=False)

    def comments_by_user(self, request, user_id):
        user = get_object_or_404(User, id=user_id)
        comments = Commentaire.objects.filter(user=user)
        data = {
            'comments': [comment.content for comment in comments]
        }
        return JsonResponse(data)


def load_modell():
    model_path = os.path.join(settings.BASE_DIR, 'api', 'user', 'vgg_19tl.model')
    return tf.keras.models.load_model(model_path)


def load_model():
    model_path = os.path.join(settings.BASE_DIR, 'api', 'user', 'potatoes_model')
    return tf.keras.models.load_model(model_path)


def read_file_as_image(file):
    data = file.read()
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))
    return image


def getToken(token):
    all_token = ActiveSession.objects.all()
    for obj in all_token:
        if obj.token == token:
            return token
    return False;


def get_user_by_token(token):
    all_token = ActiveSession.objects.all()
    for obj in all_token:
        if obj.token == token:
            return obj.user
    return False;


class ImageView(APIView):
    def post(self, request):

        # Get the image file from the request
        image_file = request.FILES.get('image', None)
        type_image = request.POST.get('type')
        token = request.POST.get('token')
        result = getToken(token)

        prediction = Prediction()

        if image_file and type_image and result != False:

            # Use context managers to open the file
            with Image.open(image_file) as img:
                # Convert the image to RGB format
                img = img.convert('RGB')

                # Check the content type of the image
                # if img.format.lower() not in ['jpeg', 'png', 'gif','jpg']:
                #     return Response({'error': 'Invalid image format'}, status=400)
                # Redimensionner l'image
                resized_image = cv2.resize(np.array(img), (128, 128))
                # Convertir l'image en un tableau numpy
                image_array = np.array(resized_image, dtype=np.float32)
                # Prétraiter l'image
                image_array /= 255.0
                image_array = np.expand_dims(image_array, axis=0)

                if type_image == 'tomatoes':
                    # Faire la prédiction
                    model = load_modell()

                    CLASS_NAMES = ["Tomato___Bacterial_spot", "Tomato___Early_blight",
                                   "Tomato___healthy", "Tomato___Late_blight",
                                   "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
                                   "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
                                   "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

                    predictions = model.predict(image_array)
                    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

                    # fonction à implémenter qui retourne la classe prédite
                    attributes = {
                        "class": predicted_class,  # test(predictions)
                        "confidence": str(np.max(predictions))
                    }

                    prediction.type = type_image
                    prediction.classe = predicted_class
                    prediction.coeff = float(str(np.max(predictions)))
                    prediction.date = date.today()
                    prediction.user = get_user_by_token(token)
                    prediction.save()
                    # return Response(attributes, status=200)
                elif type_image == "potatoes":
                    resized_image = cv2.resize(np.array(img), (256, 256))

                    # Convertir l'image en un tableau numpy
                    image_array = np.array(resized_image, dtype=np.float32)

                    # Prétraiter l'image
                    image_array /= 255.0
                    image_array = np.expand_dims(image_array, axis=0)
                    # Faire la prédiction
                    model = load_model()
                    CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
                    predictions = model.predict(image_array)
                    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                    confidence = np.max(predictions[0])
                    attributes = {
                        "class": predicted_class,  # test(predictions)
                        "confidence": float(confidence)
                    }
                    prediction = Prediction()
                    prediction.type = type_image
                    prediction.classe = predicted_class
                    prediction.coeff = float(str(np.max(predictions)))
                    prediction.date = date.today()
                    prediction.user = get_user_by_token(token)
                    prediction.save()
                    return JsonResponse(attributes)
                else:
                    attributes = {
                        "class": "Erreur de prediction",  # test(predictions)
                        "confidence": "Erreur de prediction"
                    }

                return JsonResponse(attributes)

        return JsonResponse({'error': 'Erreur réessayez autre fois !'}, status=200)

    def get(self, request):
        if request.method == "GET":
            print("test")
        return JsonResponse({'error': 'Invalid request method'})


class CommentaireView(APIView):
    def post(self, request):
        # if request.method =="POST":
        content = request.POST.get('content')
        token = request.POST.get('token')
        if content != "" and token != "":
            try:
                commentaire = Commentaire()
                commentaire.date = date.today()
                commentaire.content = content
                commentaire.user = get_user_by_token(token)
                commentaire.save()
                return JsonResponse({'success': 'Votre commentaire a été enregistré avec succès.'})
            except:
                return JsonResponse({'error': 'Votre commentaire n\'est pas enregistré, réessayez'})

    def get(self, request):
        if request.method == "GET":
            print("test")
        return JsonResponse({'error': 'Invalid request method'})


class EditProfileView(APIView):
    def post(self, request):
        if request.method == 'POST':
            token = request.POST.get('token')
            firstname = request.POST.get("firstName")
            lastname = request.POST.get("lastName")
            email = request.POST.get('email')
            address = request.POST.get('address')
            telephone = request.POST.get('telephone')
            dateNaissance = request.POST.get('date_naissance')
            if token != "":
                # user_id = get_user_by_token(token)
                all_user = User.objects.all()
                for u in all_user:
                    if u.email == email:
                        user = u
                        user.address = address
                        user.firstName = firstname
                        user.lastName = lastname
                        user.telephone = telephone
                        user.dateNaissance = dateNaissance
                        user.save()
                        user_data = {
                            "firstName": firstname,
                            "lastName": lastname,
                            "email": user.email,
                            "telephone": telephone,
                            "address": address,
                            "dateNaissance": dateNaissance
                        }
                        return JsonResponse(user_data)
            return JsonResponse({"error": "erreur au niveau de token"})


class UserProfileView(APIView):
    def post(self, request):
        if request.method == "POST":
            email = request.POST.get("email")
            all_user = User.objects.all()
            for u in all_user:
                if u.email == email:
                    data_user ={
                        "email" : u.email,
                        "firstName" : u.firstName,
                        "lastName" : u.lastName,
                        "telephone" : u.telephone,
                        "address": u.address,
                        "naissance" : u.dateNaissance
                    }
                    return JsonResponse({"success": data_user})
            return JsonResponse({"error": "error"})
