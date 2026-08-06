import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"

    allowed_domains = ["books.toscrape.com"]

    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    def parse(self, response):

        # Visit each book on the current page
        for book in response.css("article.product_pod"):

            detail_url = response.urljoin(
                book.css("h3 a::attr(href)").get()
            )

            yield response.follow(
                detail_url,
                callback=self.parse_book
            )

        # Get the current page number
        current_page = int(
            response.url.split("page-")[-1].split(".")[0]
        )

        # Continue only until page 5
        if current_page < 5:

            next_page = response.css(
                "li.next a::attr(href)"
            ).get()

            if next_page:
                yield response.follow(
                    next_page,
                    callback=self.parse
                )

    def parse_book(self, response):

        yield {

            "title": response.css(
                "div.product_main h1::text"
            ).get(),

            "category": response.css(
                "ul.breadcrumb li:nth-child(3) a::text"
            ).get(),

            "price": response.css(
                "p.price_color::text"
            ).get(),

            "rating": response.css(
                "p.star-rating::attr(class)"
            ).get().split()[-1],

            "availability": " ".join(
                text.strip()
                for text in response.css("p.instock.availability::text").getall()
                if text.strip()
            ),

            "description": response.css(
                "#product_description + p::text"
            ).get(),

            "upc": response.xpath(
                '//th[text()="UPC"]/following-sibling::td/text()'
            ).get(),

            "num_reviews": response.xpath(
                '//th[text()="Number of reviews"]/following-sibling::td/text()'
            ).get(),

            "product_url": response.url,
        }