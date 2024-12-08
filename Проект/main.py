import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tkinter import Tk, Button, Label, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webbrowser
import os
import tempfile
import pandas as pd

# Загружаем данные Titanic
titanic = sns.load_dataset("titanic")
titanic.dropna(subset=['age', 'fare'], inplace=True)

# Переименовываем классы для большей информативности
class_mapping = {
    'First': 'Пассажиры первого класса',
    'Second': 'Пассажиры второго класса',
    'Third': 'Пассажиры третьего класса'
}

titanic['class'] = titanic['class'].map(class_mapping)

# --- Tkinter окно ---
def create_main_window():
    root = Tk()
    root.title("Визуализация данных Titanic")
    root.geometry("1600x800")

    # Заголовок
    Label(root, text="Визуализация данных Titanic", font=("Arial", 16)).pack(pady=20)

    # Контейнер для графиков
    frame = Frame(root)
    frame.pack(fill="both", expand=True)

    # --- Matplotlib ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    # Добавление меток классов на гистограмму возрастов
    for pclass in titanic['class'].unique():
        ax1.hist(titanic[titanic['class'] == pclass]['age'], bins=15, alpha=0.6, label=f'{pclass}')
    ax1.set_title('Распределение возрастов по классам')
    ax1.set_xlabel('Возраст')
    ax1.set_ylabel('Частота')
    ax1.legend()

    canvas1 = FigureCanvasTkAgg(fig1, master=frame)
    canvas1.get_tk_widget().grid(row=0, column=0, padx=30, pady=20)

    # --- Seaborn ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    # Корреляционная тепловая карта Seaborn
    corr_matrix = titanic[['age', 'fare', 'sibsp', 'parch']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    ax2.set_title('Корреляционная тепловая карта')

    ax2.set_xticklabels(['Возраст', 'Стоимость\nбилета', 'Братья/\nсестры', 'Родители/\nдети'])
    ax2.set_yticklabels(['Возраст', 'Стоимость\nбилета', 'Братья/\nсестры', 'Родители/\nдети'])

    canvas2 = FigureCanvasTkAgg(fig2, master=frame)
    canvas2.get_tk_widget().grid(row=0, column=1, padx=30, pady=20)

    # --- Plotly ---
    def show_plotly():
        plotly_fig = px.scatter(
            titanic,
            x="age",
            y="fare",
            color="class",
            size="fare",
            hover_data=["sex", "embark_town"],
            labels={
                "age": "Возраст",
                "fare": "Стоимость билета"
            },
            title="Интерактивный график: Возраст vs. Стоимость билета"
        )

        temp_dir = tempfile.mkdtemp()
        plotly_file_path = os.path.join(temp_dir, "plotly_graph.html")
        plotly_fig.write_html(plotly_file_path)

        webbrowser.open(f'file://{plotly_file_path}')

    # --- Кнопка для Plotly-графика ---
    Button(root, text="Открыть интерактивный график Plotly", command=show_plotly, bg="lightblue", font=("Arial", 12)).pack(pady=20)

    root.mainloop()

# --- Основной запуск ---
if __name__ == "__main__":
    create_main_window()
