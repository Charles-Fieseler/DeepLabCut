
def xml_pretty_print(root, max_depth=1, current_depth=1):
    for child in root:
        print(child.tag)
        print(child.attrib)
        if current_depth < max_depth:
            xml_pretty_print(child, max_depth=max_depth, current_depth=current_depth+1)
