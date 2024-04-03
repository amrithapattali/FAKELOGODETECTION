from django.contrib import admin
from .models import *
# Register your models here.
from .models import CustUser

# class ComplaintAdmin(admin.ModelAdmin):
#     list_display = ['user', 'complaint_text', 'created_at']
#     list_filter = ['created_at']
#     search_fields = ['user__username', 'complaint_text']
#     readonly_fields = ['user', 'complaint_text', 'created_at']

#     def has_view_permission(self, request, obj=None):
#         # Only allow users with is_superuser permission to view complaints
#         if request.user.is_superuser:
#             return True
#         return False

#     def get_actions(self, request):
#         actions = super().get_actions(request)
#         if 'remove_store' in actions:
#             del actions['remove_store']  # Remove the default delete action
#         return actions

#     def remove_store(self, request, queryset):
#         for complaint in queryset:
#             store_complaint = complaint.complaint_text.lower().find('store') != -1
#             if store_complaint:
#                 # Remove the store or perform actions as needed
#                 # For demonstration purposes, let's just delete the complaint
#                 complaint.delete()

#     remove_store.short_description = "Remove store based on complaints"

admin.site.register(CustUser)
# admin.site.register(Complaint, ComplaintAdmin)


