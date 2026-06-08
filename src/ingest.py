import pandas as pd

class Ingestion:

    def ingest_data(self):
        users_df = pd.read_csv("data/source/users.csv")

        shopping_cart_df = pd.read_csv("data/source/shoppingcartitem.csv")

        enrollment_df = pd.read_csv("data/source/enrollment.csv")

        classes_df = pd.read_csv("data/source/classes.csv")

        sales_order_df = pd.read_csv("data/source/salesorder.csv")

        discount_df = pd.read_csv("data/source/discount.csv")

        guest_users_df = pd.read_csv("data/source/guestuser.csv")

        transfer_class_df = pd.read_csv("data/source/transferclass.csv")

        order_weight_df = pd.read_csv("data/source/order_weight.csv")

        transfer_weight_df = pd.read_csv("data/source/transfer_weight.csv")

        enroll_weight_df = pd.read_csv("data/source/enroll_weight.csv")

        duration_df = pd.read_csv("data/source/duration_bucket.csv")

        channel_df = pd.read_csv("data/source/channel_mapping.csv")

        recency_df = pd.read_csv("data/source/recency_config.csv")

        time_of_day_df = pd.read_csv("data/source/time_of_day_bucket.csv")

        cost_tier_df = pd.read_csv("data/source/cost_tier_bucket.csv")

        return (users_df, classes_df, enrollment_df, sales_order_df, shopping_cart_df,
                discount_df, guest_users_df, transfer_class_df, order_weight_df,
                transfer_weight_df, enroll_weight_df, duration_df, channel_df, recency_df, time_of_day_df, cost_tier_df)



