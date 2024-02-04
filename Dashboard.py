import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit, 
                             QLabel, QPushButton)

class TradingDashboard(QWidget):
    def __init__(self, portfolio):
        super().__init__()
        self.initUI(portfolio)
        
    def initUI(self, portfolio):
        # Set up the window
        self.setWindowTitle('Trading Portfolio Dashboard')
        layout = QVBoxLayout()

        # Add a label
        self.label = QLabel('Your portfolio:')
        layout.addWidget(self.label)

        # Add a text edit for displaying portfolio details
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(str(portfolio))
        layout.addWidget(self.text_edit)

        # Add a button to refresh or submit changes
        self.button = QPushButton('Refresh')
        self.button.clicked.connect(self.refresh_portfolio)
        layout.addWidget(self.button)

        # Set the layout
        self.setLayout(layout)

    def refresh_portfolio(self):
        # Refresh or process the portfolio data here
        portfolio_data = self.text_edit.toPlainText()
        self.label.setText(f'Your portfolio: {portfolio_data}')

# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TradingDashboard(portfolio="Initial portfolio data")
    ex.show()
    sys.exit(app.exec_())

