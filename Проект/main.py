from tkinter import Tk, Label, Frame
from tkinter.ttk import Combobox
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

# --- Plotly функции ---
def show_plotly_scatter():
    scatter_fig = px.scatter(
        titanic,
        x="age",
        y="fare",
        color="class",
        size="fare",
        hover_data=["sex", "embark_town"],
        labels={"age": "Возраст", "fare": "Стоимость билета"},
        title="Интерактивный график: Возраст vs. Стоимость билета"
    )
    show_plotly_figure(scatter_fig)

def show_plotly_pie():
    pie_fig = px.pie(
        titanic,
        names='sex',
        title='Распределение пассажиров по полу',
        color_discrete_sequence=['blue', 'pink']
    )
    show_plotly_figure(pie_fig)

def show_plotly_3d():
    three_d_fig = px.scatter_3d(
        titanic,
        x="age",
        y="fare",
        z="pclass",
        color="class",
        title="Трехмерный график: Возраст, Стоимость билета и Класс"
    )
    show_plotly_figure(three_d_fig)

def show_plotly_figure(plotly_fig):
    temp_dir = tempfile.mkdtemp()
    plotly_file_path = os.path.join(temp_dir, "plotly_graph.html")
    plotly_fig.write_html(plotly_file_path)
    webbrowser.open(f'file://{plotly_file_path}')

# --- Tkinter окно ---
def create_main_window():
    root = Tk()
    root.title("Визуализация данных Titanic")
    root.geometry("1600x1200")

    # Заголовок
    Label(root, text="Визуализация данных Titanic", font=("Arial", 16)).pack(pady=20)

    # Контейнер для графиков
    frame = Frame(root)
    frame.pack(fill="both", expand=True)

    # --- Matplotlib графики ---
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for pclass in titanic['class'].unique():
        ax1.hist(titanic[titanic['class'] == pclass]['age'], bins=15, alpha=0.6, label=f'{pclass}')
    ax1.set_title('Распределение возрастов по классам')
    ax1.set_xlabel('Возраст')
    ax1.set_ylabel('Частота')
    ax1.legend()

    canvas1 = FigureCanvasTkAgg(fig1, master=frame)
    canvas1.get_tk_widget().grid(row=0, column=0, padx=20, pady=20)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    survived_counts = titanic['survived'].value_counts()
    labels = ['Погибшие', 'Выжившие']
    ax2.pie(survived_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['salmon', 'lightgreen'])
    ax2.set_title('Распределение выживших и погибших')

    canvas2 = FigureCanvasTkAgg(fig2, master=frame)
    canvas2.get_tk_widget().grid(row=0, column=1, padx=20, pady=20)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    age_bins = pd.cut(titanic['age'], bins=5)
    avg_fare_by_age = titanic.groupby(age_bins)['fare'].mean()
    ax3.plot(avg_fare_by_age.index.astype(str), avg_fare_by_age, marker='o', linestyle='-', color='blue')
    ax3.set_title('Средняя стоимость билета по возрастным группам')
    ax3.set_xlabel('Возрастная группа')
    ax3.set_ylabel('Средняя стоимость билета')

    canvas3 = FigureCanvasTkAgg(fig3, master=frame)
    canvas3.get_tk_widget().grid(row=0, column=2, padx=20, pady=20)

    # --- Seaborn графики ---
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    corr_matrix = titanic[['age', 'fare', 'sibsp', 'parch']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4)
    ax4.set_title('Корреляционная тепловая карта')

    canvas4 = FigureCanvasTkAgg(fig4, master=frame)
    canvas4.get_tk_widget().grid(row=1, column=0, padx=20, pady=20)

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=titanic, x='fare', hue='class', fill=True, ax=ax5, alpha=0.6)
    ax5.set_title('Распределение стоимости билетов по классам')

    canvas5 = FigureCanvasTkAgg(fig5, master=frame)
    canvas5.get_tk_widget().grid(row=1, column=1, padx=20, pady=20)

    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=titanic, x='class', y='fare', ax=ax6)
    ax6.set_title('Стоимость билетов по классам')
    ax6.set_xlabel('Класс')
    ax6.set_ylabel('Стоимость билета')

    canvas6 = FigureCanvasTkAgg(fig6, master=frame)
    canvas6.get_tk_widget().grid(row=1, column=2, padx=20, pady=20)

    # --- Выпадающий список для Plotly-графиков ---
    dropdown_frame = Frame(root)
    dropdown_frame.pack(pady=10)

    Label(dropdown_frame, text="Выберите график:", font=("Arial", 12)).pack(side="left", padx=5)

    options = ["Scatter Plot", "Pie Chart", "3D Scatter Plot"]
    combobox = Combobox(dropdown_frame, values=options, state="readonly", font=("Arial", 12), width=20)
    combobox.pack(side="left", padx=5)

    def handle_selection(event):
        selected_option = combobox.get()
        if selected_option == "Scatter Plot":
            show_plotly_scatter()
        elif selected_option == "Pie Chart":
            show_plotly_pie()
        elif selected_option == "3D Scatter Plot":
            show_plotly_3d()

    combobox.bind("<<ComboboxSelected>>", handle_selection)

    root.mainloop()

# --- Основной запуск ---
if __name__ == "__main__":
    create_main_window()
