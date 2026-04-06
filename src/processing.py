import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from phik import resources, report

from sqlalchemy import create_engine


class DataProcessor:
    def __init__(self):
        """Инициализация класса. Загрузка доступов и подготовка переменных."""
        load_dotenv()

        self.tickets = None
        self.profiles = None

    def _get_db_engine(self):
        """Приватный метод для создания подключения к БД."""
        connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
            os.getenv('DB_USER'),
            os.getenv('DB_PWD'),
            os.getenv('DB_HOST'),
            os.getenv('DB_PORT'),
            os.getenv('DB_NAME'),
        )
        return create_engine(connection_string)

    def load_data(self):
        """Загрузка данных."""
        print("[INFO] Подключение к БД и выгрузка данных...")
        engine = self._get_db_engine()

        query = '''
            WITH
            -- Собираем данные по заказам с мобильных и веб устройств:
            orders AS (
                SELECT
                    -- Информация о пользователе:
                    user_id,
                    device_type_canonical,
                    -- Информация о заказе:
                    order_id, -- Номер заказа
                    created_dt_msk AS order_dt, -- Дата заказа
                    created_ts_msk AS order_ts, -- Время заказа
                    currency_code, -- Код валюты
                    revenue, -- Выручка сервиса
                    tickets_count, -- Кол-во билетов
                    -- Информация о мероприятии:
                    event_id,
                    service_name,
                    -- Найдем время между заказами для каждого пользователя:
                    created_dt_msk::date - LAG(created_dt_msk) OVER(PARTITION BY user_id ORDER BY created_dt_msk)::date AS days_since_prev
                FROM afisha.purchases
                WHERE
                    -- Фильтруем тип устройства
                    device_type_canonical IN ('mobile','desktop')
            ),
            -- Собираем информацию по событиям:
                events AS (
                SELECT
                    -- Выгружаем данные таблицы events:
                    e.event_id,
                    e.event_name_code AS event_name,
                    e.event_type_main,
                    -- Выгружаем информацию о городе и регионе:
                    r.region_name,
                    c.city_name
                FROM afisha.events AS e
                LEFT JOIN afisha.venues AS v USING(venue_id)
                LEFT JOIN afisha.city AS c USING(city_id)
                LEFT JOIN afisha.regions AS r USING(region_id)
                WHERE e.event_type_main != 'фильм'
            )
            -- Создаем итоговую выгрузку:
            SELECT
                orders.*,
                event_name,
                event_type_main,
                region_name,
                city_name
            FROM orders INNER JOIN events USING (event_id)
            ORDER BY user_id, order_dt;
            '''

        with engine.connect() as connection:
            self.tickets = pd.read_sql_query(query, con=connection)

        print("[INFO] Данные успешно выгружены.")

    def convert_currency(self, tenge_file_path='data/final_tickets_tenge_df.csv'):
        """Конвертация тенге в рубли."""
        print("\n[INFO] Конвертация валюты. Загрузка справочника курсов...")

        tenge = pd.read_csv(tenge_file_path)
        tenge['data'] = pd.to_datetime(tenge['data'])
        tenge_map = tenge.set_index('data')['curs']

        self.tickets['curs'] = self.tickets['order_dt'].map(tenge_map)

        is_kzt = self.tickets['currency_code'] == 'kzt'
        self.tickets['revenue_rub'] = np.where(
            is_kzt,
            self.tickets['revenue'] * self.tickets['curs'] / 100,
            self.tickets['revenue']
        )

        print("[INFO] Конвертация завершена. Добавлен столбец 'revenue_rub'.")

    def preprocess_data(self):
        """Очистка данных, удаление выбросов и дубликатов, смена типов."""
        print("\n[INFO] Начало предобработки данных...")
        initial_len = len(self.tickets)

        rub99 = np.percentile(self.tickets['revenue_rub'].dropna(), 99)
        print(f"[INFO] Значение 99 перцентиля для заказов: {rub99:.0f} руб.")

        self.tickets = self.tickets[
            (self.tickets['revenue_rub'] < rub99) &
            (self.tickets['revenue_rub'] > 0)
            ].copy()

        columns_to_dupl = ['user_id', 'order_ts', 'device_type_canonical',
                           'service_name', 'event_id', 'revenue']

        duplicates_count = self.tickets[columns_to_dupl].duplicated().sum()
        print(f"[INFO] Найдено полных дубликатов по выбранным столбцам: {duplicates_count}")

        self.tickets = self.tickets.drop_duplicates(subset=columns_to_dupl)

        int_cols = ['order_id', 'event_id', 'tickets_count']
        float_cols = ['revenue', 'revenue_rub', 'days_since_prev']

        for col in int_cols:
            if col in self.tickets.columns:
                self.tickets[col] = pd.to_numeric(self.tickets[col], downcast='integer')

        for col in float_cols:
            if col in self.tickets.columns:
                self.tickets[col] = pd.to_numeric(self.tickets[col], downcast='float')

        final_len = len(self.tickets)
        filter_percent = 1 - (final_len / initial_len)
        print(f"[INFO] Предобработка завершена. Отфильтровано {filter_percent:.2%} данных.")

    def generate_eda_reports(self):
        """Разведочный анализ (вывод статистик)."""
        print("\n[INFO] Статистика по очищенным данным")
        print("Статистические показатели столбца revenue_rub:")
        print(self.tickets['revenue_rub'].describe())

        print("\nАгрегация количества билетов по коду валюты:")
        agg_df = self.tickets.groupby('currency_code').agg(
            tickets_min=('tickets_count', 'min'),
            tickets_max=('tickets_count', 'max'),
            tickets_avg=('tickets_count', 'mean'),
            tickets_p95=('tickets_count', lambda x: np.percentile(x.dropna(), 95)),
            tickets_p99=('tickets_count', lambda x: np.percentile(x.dropna(), 99)),
            tickets_var=('tickets_count', 'var')
        )
        print(agg_df.to_string())

    def create_user_profiles(self):
        """Создание профилей пользователей (агрегация на уровне user_id)."""
        print("\n[INFO] Создание профилей пользователей...")

        self.profiles = (
            self.tickets
            .sort_values(by='order_ts')
            .groupby('user_id')
            .agg(
                first_order_dt=('order_dt', 'min'),
                last_order_dt=('order_dt', 'max'),
                first_device=('device_type_canonical', 'first'),
                first_region_name=('region_name', 'first'),
                first_event_type=('event_type_main', 'first'),
                total_orders=('order_id', 'nunique'),
                avg_revenue_rub=('revenue_rub', 'mean'),
                avg_days_since_prev=('days_since_prev', 'mean')
            )
            .assign(
                is_two=lambda x: x['total_orders'] >= 2,
                avg_days=lambda x: (x['last_order_dt'] - x['first_order_dt']).dt.days / (x['total_orders'] - 1)
            )
            .reset_index()
        )
        print(f"[INFO] Успешно сформировано {len(self.profiles)} профилей.")

    def analyze_and_filter_profiles(self):
        """Анализ аномалий в профилях и фильтрация выбросов."""
        print("\n[INFO] Анализ профилей до фильтрации")
        initial_users = self.profiles['user_id'].nunique()
        print(f"Общее число пользователей: {initial_users}")
        print(f"Доля пользователей совершивших 2 и более заказа: {self.profiles['is_two'].mean():.2%}")

        print("\nСтатистика (Изначальная):")
        print(self.profiles[['total_orders', 'avg_revenue_rub', 'avg_days_since_prev']]
              .describe(percentiles=[.01, .25, .5, .75, .95, .99]).T.to_string())

        orders99 = np.percentile(self.profiles['total_orders'].dropna(), 99)
        print(f"\n[INFO] Значение 99 перцентиля для количества заказов: {orders99:.0f}")

        self.profiles = self.profiles[
            (self.profiles['total_orders'] < orders99) &
            (self.profiles['avg_revenue_rub'] > 0)
            ].copy()

        final_users = self.profiles['user_id'].nunique()
        filtered_pct = 1 - (final_users / initial_users)

        print("\n[INFO] Анализ профилей после фильтрации")
        print(f"Осталось пользователей: {final_users}")
        print(f"Было отфильтровано: {filtered_pct:.2%} пользователей")
        print(f"Доля пользователей совершивших 2 и более заказа: {self.profiles['is_two'].mean():.2%}")

        print("\nСтатистика (Очищенная):")
        print(self.profiles[['total_orders', 'avg_revenue_rub', 'avg_days_since_prev']]
              .describe(percentiles=[.01, .25, .5, .75, .95, .99]).T.to_string())

    def analyze_user_segments(self):
        """Изучение распределения пользователей по признакам."""
        print("\n[INFO] Анализ сегментов пользователей")

        segments = ['first_device', 'first_event_type', 'first_region_name']

        for segment in segments:
            print(f"\nРаспределение пользователей по признаку: {segment}")

            df_grouped = (
                self.profiles.groupby(segment)
                .agg(total_users=('user_id', 'nunique'))
                .assign(users_share=lambda x: x['total_users'] / x['total_users'].sum())
                .sort_values(by='users_share', ascending=False)
            )

            print(df_grouped.to_string())

    def analyze_returning_users(self):
        """Анализ возвратов пользователей (Retetion rate)."""
        print("\n[INFO] Анализ возвратов (повторных заказов)")
        segments = ['first_device', 'first_event_type', 'first_region_name']

        overall_mean_share = self.profiles['is_two'].mean()

        for segment in segments:
            print(f"\nОтток/Возврат по признаку: {segment}")

            res = (
                self.profiles.groupby(segment)
                .agg(
                    total_users=('user_id', 'nunique'),
                    is_two_share=('is_two', 'mean')
                )
                .nlargest(10, 'total_users')
                .sort_values(by='is_two_share', ascending=False)
            )

            res['is_higher_mean'] = res['is_two_share'] > overall_mean_share

            share = res["total_users"].sum() / len(self.profiles)
            print(f"Выборка топ-10 сегментов охватывает {share:.1%} пользователей.")
            print(res.to_string())

    def analyze_weekday_impact(self):
        """Влияние дня недели первой покупки на возвращаемость."""
        print("\n[INFO] Влияние дня недели первого заказа")

        self.profiles['weekday'] = pd.to_datetime(self.profiles['first_order_dt']).dt.day_name()

        res = (
            self.profiles.groupby('weekday')
            .agg(
                total_users=('is_two', 'count'),
                is_two_share=('is_two', 'mean')
            )
        )

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        res = res.reindex(days_order)

        print("Статистика по дням недели:")
        print(res.to_string())

    def calculate_correlations(self):
        """Корреляционный анализ Phi_K с исходными данными и сегментами."""
        print("\n[INFO] Корреляционный анализ (Phi_k)")

        print("[INFO] Расчет первичной корреляционной матрицы...")
        cols_initial = ['first_device', 'first_region_name', 'first_event_type',
                        'weekday', 'avg_revenue_rub', 'avg_days_since_prev',
                        'avg_days', 'total_orders']

        df_corr1 = self.profiles[cols_initial].dropna()
        intervals1 = ['avg_revenue_rub', 'avg_days_since_prev', 'avg_days', 'total_orders']

        corr_matrix1 = df_corr1.phik_matrix(interval_cols=intervals1)

        res_total = (
            corr_matrix1.loc[corr_matrix1.index != 'total_orders'][['total_orders']]
            .sort_values(by='total_orders', ascending=False)
        )
        print("Корреляция с общим числом заказов (total_orders):")
        print(res_total.to_string())

        print("\n[INFO] Сегментирование пользователей по количеству заказов...")

        conditions = [
            self.profiles['total_orders'] == 1,
            self.profiles['total_orders'] == 2,
            self.profiles['total_orders'].between(3, 4),
            self.profiles['total_orders'] >= 5
        ]

        choices = [
            '1 заказ',
            '2 заказа',
            'от 3 до 4 заказов',
            'от 5 и более'
        ]

        self.profiles['orders_segment'] = np.select(conditions, choices)

        print("Распределение по сегментам:")
        print(self.profiles['orders_segment'].value_counts().to_string())

        print("\n[INFO] Расчет корреляции для сегментов...")

        cols_seg = ['first_device', 'first_region_name', 'first_event_type',
                    'weekday', 'avg_revenue_rub', 'avg_days_since_prev',
                    'avg_days', 'orders_segment']

        df_corr2 = self.profiles[cols_seg].dropna()
        intervals2 = ['avg_revenue_rub', 'avg_days_since_prev', 'avg_days']

        corr_matrix2 = df_corr2.phik_matrix(interval_cols=intervals2)

        orders_segment_res = (
            corr_matrix2.loc[corr_matrix2.index != 'orders_segment'][['orders_segment']]
            .sort_values(by='orders_segment', ascending=False)
        )
        print("Корреляция с сегментами (orders_segment):")
        print(orders_segment_res.to_string())
