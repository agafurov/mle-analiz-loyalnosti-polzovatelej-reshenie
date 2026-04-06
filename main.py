from src.processing import DataProcessor


def main():
    print("[INFO] Запуск пайплайна")

    processor = DataProcessor()

    processor.load_data()
    processor.convert_currency()
    processor.preprocess_data()

    processor.create_user_profiles()
    processor.analyze_and_filter_profiles()

    processor.generate_eda_reports()

    processor.analyze_user_segments()
    processor.analyze_returning_users()
    processor.analyze_weekday_impact()
    processor.calculate_correlations()

    print("\n[INFO] Пайплайн успешно завершен")


if __name__ == '__main__':
    main()
