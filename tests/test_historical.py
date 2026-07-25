import unittest

from scrapper.historical.results import parse_results


class HistoricalTests(unittest.TestCase):
    def test_result_archive_parser(self):
        html = """<table><tr><th>Pla.</th><th>Horse No.</th></tr>
        <tr><td>1</td><td>4</td><td>SIGHT DREAMER (J542)</td><td>A Atzeni</td>
        <td>J Size</td><td>134</td><td>1306</td><td>8</td><td>---</td>
        <td>5 4 1 1</td><td>1:21.99</td><td>8.4</td></tr></table>"""
        rows = parse_results(html)
        self.assertEqual("J542", rows[0].horse_code)
        self.assertEqual(8.4, rows[0].win_odds)

    def test_dead_heat_placing_is_preserved(self):
        html = """<table>
        <tr><td>1 DH</td><td>4</td><td>FIRST (J542)</td><td>A Atzeni</td>
        <td>J Size</td><td>134</td><td>1306</td><td>8</td><td>---</td>
        <td>5 4 1 1</td><td>1:21.99</td><td>8.4</td></tr>
        <tr><td>1 DH</td><td>7</td><td>SECOND (J543)</td><td>Z Purton</td>
        <td>C Fownes</td><td>128</td><td>1100</td><td>2</td><td>---</td>
        <td>2 2 2 1</td><td>1:21.99</td><td>4.2</td></tr></table>"""
        rows = parse_results(html)
        self.assertEqual([1, 1], [row.place for row in rows])

    def test_legacy_horse_code_prefix_is_removed(self):
        html = """<table><tr><td>1</td><td>5</td><td>BRAVE HEART (CK277)</td>
        <td>A Atzeni</td><td>J Size</td><td>134</td><td>1306</td><td>8</td>
        <td>---</td><td>5 4 1 1</td><td>1:21.99</td><td>8.4</td></tr></table>"""
        rows = parse_results(html)
        self.assertEqual("K277", rows[0].horse_code)
        self.assertEqual("BRAVE HEART", rows[0].horse_name)


if __name__ == "__main__":
    unittest.main()
