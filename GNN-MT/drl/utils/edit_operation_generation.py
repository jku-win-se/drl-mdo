from pyecore.ecore import EReference, EClass
from pyecore.resources import ResourceSet, URI

SEP_INV = "_inv"
SPECIAL = "<special>"


def get_edge_node_dic(path_metamodel, root_element):
    rset = ResourceSet()
    # load model
    resource = rset.get_resource(URI(path_metamodel))
    # get list of all elements of the model
    list_elements = []
    for root in resource.contents:
        # list_elements.append(root)
        list_elements = list_elements + list(root.eAllContents())

    # Find the 'EClass' instance with name "StackModel" and validate its EReferences
    root_elements = [
        elem
        for elem in list_elements
        if isinstance(elem, EClass) and elem.name == root_element
    ]

    for r_model_element in root_elements:
        # Check if the element only contains "containment" EReferences
        if all(ref.containment for ref in r_model_element.eReferences):
            # Remove references of the element
            for ref in r_model_element.eReferences:
                list_elements.remove(ref)
            # Remove the element itself
            list_elements.remove(r_model_element)

            # Remove its containment EReferences from other elements
            for elem in [el for el in list_elements if isinstance(el, EClass)]:
                for ref in elem.eReferences:
                    if ref.eType == r_model_element:
                        elem.eReferences.remove(ref)

    # Make edge and node label extraction deterministic
    edge_labels = sorted(
        list(set([e.name for e in list_elements if isinstance(e, EReference)]))
    )
    node_labels = sorted(
        list(set([c.name for c in list_elements if isinstance(c, EClass)]))
    )

    edge_labels += [e + SEP_INV for e in edge_labels]
    edge_labels += [SPECIAL]

    edge_dict = {y: x for x, y in enumerate(edge_labels)}
    node_dict = {y: x for x, y in enumerate(node_labels)}

    # Build subclass lookup dictionary
    class_hierarch_dict = {label: [] for label in node_labels}
    class_ancestry_dict = {label: [] for label in node_labels}

    # Map class names to EClass instances - ensure deterministic processing
    name_to_class = {cls.name: cls for cls in list_elements if isinstance(cls, EClass)}

    # Build reverse and forward inheritance graphs with deterministic iteration
    subclass_graph = {cls_name: [] for cls_name in sorted(name_to_class.keys())}
    superclass_graph = {cls_name: [] for cls_name in sorted(name_to_class.keys())}

    # Process classes in sorted order for deterministic results
    for cls_name in sorted(name_to_class.keys()):
        cls = name_to_class[cls_name]
        for supercls in [st for st in cls.eSuperTypes if st.name != "NamedElement"]:
            subclass_graph[supercls.name].append(cls.name)
            superclass_graph[cls_name].append(supercls.name)

    # Recursive function to get all subclasses (descendants)
    def collect_all_subclasses(class_name, seen=None):
        if seen is None:
            seen = set()
        # Sort for deterministic iteration order
        for sub in sorted(subclass_graph.get(class_name, [])):
            if sub not in seen:
                seen.add(sub)
                collect_all_subclasses(sub, seen)
        return seen

    # Recursive function to get all superclasses (ancestors)
    def collect_all_superclasses(class_name, seen=None):
        if seen is None:
            seen = set()
        # Sort for deterministic iteration order
        for sup in sorted(superclass_graph.get(class_name, [])):
            if sup not in seen:
                seen.add(sup)
                collect_all_superclasses(sup, seen)
        return seen

    # Build both dictionaries
    for cls_name in node_labels:
        cls_index = node_dict[cls_name]

        # Fill subclass hierarchy - sort for deterministic order
        subclass_names = sorted(collect_all_subclasses(cls_name))
        subclass_indices = [node_dict[name] for name in subclass_names]
        class_hierarch_dict[cls_name] = [cls_index] + subclass_indices

        # Fill superclass ancestry - sort for deterministic order
        superclass_names = sorted(collect_all_superclasses(cls_name))
        superclass_indices = [node_dict[name] for name in superclass_names]
        class_ancestry_dict[cls_name] = [cls_index] + superclass_indices

    return edge_dict, node_dict, class_hierarch_dict, class_ancestry_dict
