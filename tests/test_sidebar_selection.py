from rlocr.sidebar_selection import select_sidebar_pages, sidebar_witness


def example(index=0):
    def block(name,box,text):
        x0,y0,x1,y1 = box
        return {"anno_id":name,"category_type":"text_block","order":len(name),
                "text":text,"poly":[x0,y0,x1,y0,x1,y1,x0,y1],"ignore":False}
    return {"page_info":{"width":1000,"height":1000,"image_path":f"p{index}.png",
                         "page_attribute":{"language":"english","data_source":"magazine","layout":"other_layout"}},
            "layout_dets":[block("side",[20,100,250,300],"a"*40),
                           block("main",[350,100,950,600],"b"*100),
                           block("third",[350,650,950,800],"c"*100)]}


def test_frozen_geometric_proxy_and_failure_cases():
    page=example()
    assert sidebar_witness(page)["sidebar_anno_id"] == "side"
    page["layout_dets"][0]["ignore"]=True
    assert sidebar_witness(page) is None
    page=example()
    page["layout_dets"][0]["text"]="a "*39
    assert sidebar_witness(page) is None
    page=example()
    page["layout_dets"][1]["poly"]=[350,400,950,400,950,600,350,600]
    assert sidebar_witness(page) is None


def test_minimum_pool_is_enforced_without_relaxation():
    pages=[example(i) for i in range(79)]
    result=select_sidebar_pages(pages,set())
    assert result["qualifying_count"] == 79
    assert result["selected_pages"] == []
    assert result["status"] == "stopped_insufficient_qualifying_pages"


def test_exclusions_fixed_seed_and_no_outcome_dependency():
    pages=[example(i) for i in range(105)]
    original=select_sidebar_pages(pages,{"images/p0.png"})
    assert len(original["selected_pages"]) == 100
    assert all(row["image_path"] != "images/p0.png" for row in original["selected_pages"])
    for page in pages:
        page["native_tau"]=0
        page["gat_tau"]=1
    assert select_sidebar_pages(pages,{"images/p0.png"}) == original
