import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import plotly.graph_objs as go
from PyQt5.QtWebEngineWidgets import QWebEngineView
from plotly.offline import plot

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interface graphique PID")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.webview = QWebEngineView()
        layout.addWidget(self.webview)

        button = QPushButton("Tracer Courbes")
        button.clicked.connect(self.plot_curves)
        layout.addWidget(button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def plot_curves(self):
        # Exemple de données pour les courbes
        x = [0, 1, 2, 3, 4, 5]
        y1 = [0, 1, 2, 3, 4, 5]
        y2 = [5, 4, 3, 2, 1, 0]

        # Créer les traces
        trace1 = go.Scatter(x=x, y=y1, mode='lines', name='Courbe 1')
        trace2 = go.Scatter(x=x, y=y2, mode='lines', name='Courbe 2')

        # Créer la figure
        fig = go.Figure([trace1, trace2])

        # Générer le code HTML pour le graphique Plotly
        plot_html = plot(fig, include_plotlyjs=False, output_type='div')

        # Afficher le graphique dans le WebView
        self.webview.setHtml(plot_html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
