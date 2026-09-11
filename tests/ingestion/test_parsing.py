from src.ingestion.parsing import html_to_text


def test_html_to_text_strips_scripts_and_styles() -> None:
    html = """
    <html>
      <head><style>.x{color:red}</style></head>
      <body>
        <script>trackEvent();</script>
        <h1>Item 1. Business</h1>
        <p>We operate nuclear power plants.</p>
      </body>
    </html>
    """
    text = html_to_text(html)
    assert "Item 1. Business" in text
    assert "We operate nuclear power plants." in text
    assert "trackEvent" not in text
    assert "color:red" not in text


def test_html_to_text_collapses_blank_lines() -> None:
    html = "<p>Para one.</p>\n\n\n\n<p>Para two.</p>"
    text = html_to_text(html)
    assert "\n\n\n" not in text
