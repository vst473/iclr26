import scrapy
from scrapy.http import FormRequest, Request
from scrapy.selector import Selector
import time, os



DOWNLOAD_DIR = "NDLI_HIGH"
if not os.path.exists(DOWNLOAD_DIR):
    os.mkdir(DOWNLOAD_DIR)

class NDLI(scrapy.Spider):
    name = "NDLI"
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    custom_settings = {
        "CONCURRENT_REQUESTS": 100,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 100,
        "DOWNLOAD_TIMEOUT": 10000,
        "DOWNLOAD_MAXSIZE": 0,
    }

    def __init__(self):
        self.lang_code = "hin"
        self.source = "Punjab School Education Board"
        super().__init__()

    def start_requests(self):
        yield Request(url="https://ndl.iitkgp.ac.in")

    def parse(self, response):
        page = response.meta.get("x_page", None)
        yield Request(
            url="https://ndl.iitkgp.ac.in/ajax/doc-search.php",
            method="POST",
            headers={
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Content-Type": "text/plain",
                "Sec-Fetch-Site": "same-origin",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
                "X-Requested-With": "XMLHttpRequest",
                "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            },
            body='{"pageToken":' + (f'"{page}"' if page else 'null') + ',"filters":{"resourceType":["010000/010100"]},"last":null,"domain":"se","template":"list","field":"resourceType"}',
            priority=-1,
            callback=self.get_links,
        )


    def get_links(self, response):
        data = response.json()
        
        
        print(f"Page Token: {data['pageToken']}"
              f" | Total Docs: {len(data['docs'])} ")
        
        for doc in data["docs"]:
            # Create a selector from the HTML content
            doc_selector = Selector(text=doc)
            
            # Extract the URL from the card title link
            url = doc_selector.css("a.card-title.h5.link-primary::attr(href)").get()
            title = doc_selector.css("a.card-title.h5.link-primary").xpath('text()').get().strip()
            
            
            
            if url:
                yield Request(url=url, callback=self.fetch_link, meta={
                    "title": title,
                    "metadata": {"title": title}
                })
        
        if data.get("pageToken"):
            time.sleep(1)
            yield Request(
                url="https://ndl.iitkgp.ac.in/ajax/doc-search.php",
                method="POST",
                headers={
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain",
                    "Sec-Fetch-Site": "same-origin",
                    "User-Agent": self.user_agent,
                    "X-Requested-With": "XMLHttpRequest",
                    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"',
                },
                body='{"pageToken":"' + data["pageToken"] + '","filters":{"resourceType":["010000/010100"]},"last":null,"domain":"se","template":"list","field":"resourceType"}',
                priority=-1,
                callback=self.get_links,
            )
    
    
    def fetch_link(self, response):

        doc_id = response.url.split('/')[-1].split('?')[0]
        x_path = f"//*[contains(@id, '{doc_id}')]//a"
        response_data = response.xpath(x_path)

        
        # Extract all metadata using the separate function
        metadata = self.extract_metadata(response)
        for resp in response_data:
             
            url = resp.attrib['href']
            
            if url[0] == '#':
                url = response.url
            
            pdf_name = resp.xpath('text()').get().strip()
            print(f"Found PDF Name: {pdf_name} | URL: {url}")
            if not url:
                print(f"Skipping empty URL in {response.url}")
                
            if not url.startswith("http"):
                url = "http://ndl.iitkgp.ac.in" + url

            url_m = url.split("?")[0].replace("http://ndl.iitkgp.ac.in/", "").split("/")
            domain = url_m[0].split('_')[0]
            pdf_p = '/'.join(url_m[1:])
            url_m = f"http://ndl.iitkgp.ac.in/module-viewer/viewer.php?id={pdf_p}&domain={domain}"
            print(f"Processing URL: {url}")
            yield Request(url=url_m, callback=self.fetch_php_link,
                            headers={
                                "User-Agent": self.user_agent,
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
                                "Referer": url}
                            , meta={"title": response.meta.get("title"), 
                                    "s_type": "PHP_VIEWER", 
                                    "pdf_name": pdf_name,
                                    "metadata": metadata}
                            )
  
    
    
 
    def fetch_php_link(self, response):
        pdf_viewer_pattern = response.xpath('//script[contains(text(), "PDFViewerApplication.open")]/text()').get()
        if pdf_viewer_pattern:
            import re
            url_match = re.search(r'PDFViewerApplication\.open\(\{"url":"([^"]+)"\}\);', pdf_viewer_pattern)
            if url_match:
                pdf_url = url_match.group(1)
                s_type = "PDF_VIEWER_APP"
                full_url = "https://ndl.iitkgp.ac.in" + pdf_url if not pdf_url.startswith('http') else pdf_url
                print(f"Found PDF URL via PDFViewerApplication: {full_url}")
                yield Request(
                    url=full_url,
                    meta={"s_type": s_type, **response.meta},
                    callback=self.fetch_pdf_file,
                )
                return
        # external links (NCERT)
        if self.source == "NCERT":
            data = response.css("a.btn.btn-raised.btn-primary[target]").attrib["href"]
            data = f'https://ncert.nic.in/textbook/pdf/{data.split("?",1)[-1].split("=",1)[0]}{data.split("?",1)[-1].split("=",1)[1].split("-",1)[0]}.pdf'
            s_type = "EXTERNAL-NCERT"
            yield Request(
                url=data,
                meta={"s_type": s_type},
                callback=self.fetch_pdf_file,
            )
            return
        url = response.xpath("//a").attrib['href']
        # data = url.attrib["src"]
        # assests link (directly pdf)
        if url[-3:] == "pdf":
             yield Request(
                    url=url,
                    meta={"s_type": s_type, **response.meta},
                    callback=self.fetch_pdf_file,
                )
        # unhandled link (this is unhandled pls look at log file)
        else:
            s_type = "UNHANDLED"
            n_callback = None
            yield {
                "url": response.url,
                "scraped": False,
                "location": "",
                "message": f"{s_type=} {data=}",
            }
        

    def fetch_pdf_link(self, response):
        start = response.body.find(b"defaultUrl") + 14
        end = response.body.find(b");", start) - 1
        data = response.body[start:end].decode()
        yield Request(
            url="https://ndl.iitkgp.ac.in" + data,
            meta=response.meta,
            callback=self.fetch_pdf_file,
        )

    def fetch_ncert_textbooks_link(self, response):
        start = response.body.find(b"defaultUrl") + 14
        end = response.body.find(b");", start) - 1
        data = response.body[start:end].decode()
        yield Request(
            url="https://ndl.iitkgp.ac.in" + data,
            meta=response.meta,
            callback=self.fetch_pdf_file,
        )

    def fetch_pdf_file(self, response):
        s_type = response.meta.get("s_type", "-")
        if response.body[:5] != b"%PDF-":
            yield {
                "url": response.url,
                "scraped": False,
                "location": "-",
                "message": f"not a pdf file | {s_type=}",
            }
            return
        if response.meta.get("pdf_name"):
            filename = response.meta.get("pdf_name")
        else:
            
            filename = response.headers.get("Content-Disposition")
            
            if filename is None:
                filename = response.url.split("/")[-1]
            else:
                filename = filename.decode().lower().split('"')
                filename = [x for x in filename if ".pdf" in x][0]
        dir = DOWNLOAD_DIR + "/" +response.meta['metadata'].get("Content Provider", "Unknown-Content-Provider") + '/'+ response.meta['metadata'].get("Subject", "") + "/"+ response.meta['metadata'].get("Education Level", "")
        filename = f"{dir}/{filename}"
        
        os.makedirs(dir, exist_ok=True)
        
        if not filename.endswith(".pdf"):
            filename = filename + ".pdf"
        with open(filename, "wb") as fwrite:
            fwrite.write(response.body)
        
        yield {
            "url": response.url,
            "scraped": True,
            "location": filename,
            "message": f"{s_type=}",
            "meta": response.meta,
            "metadata": response.meta.get("metadata", {}),
        }
    
    def extract_metadata(self, doc_selector):
        """Extract all metadata from the HTML content"""
        metadata = {}
        
        # Extract basic metadata from table rows
        metadata_rows = doc_selector.css("table.table.table-sm tr")
        for row in metadata_rows:
            th = row.css("th::text").get()
            td = row.css("td::text").get()
            if th and td:
                metadata[th.strip()] = td.strip()
        
        # Extract subject keywords separately as they contain special formatting
        subject_keywords = doc_selector.css("table.table.table-sm tr:contains('Subject Keyword') td::text").getall()
        if subject_keywords:
            # Join all text parts and clean up
            keywords = ' '.join(subject_keywords).strip()
            metadata['Subject Keyword'] = keywords
        
        # Extract educational use, educational role (they also have special formatting)
        educational_use = doc_selector.css("table.table.table-sm tr:contains('Educational Use') td::text").getall()
        if educational_use:
            metadata['Educational Use'] = ' '.join(educational_use).strip()
            
        educational_role = doc_selector.css("table.table.table-sm tr:contains('Educational Role') td::text").getall()
        if educational_role:
            metadata['Educational Role'] = ' '.join(educational_role).strip()
        
        return metadata
    


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    
    process = CrawlerProcess(get_project_settings())
    process.crawl(NDLI)
    process.start()


