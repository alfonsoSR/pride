{% if obj.display %}
   {% if is_own_page %}
API Reference
================

.. py:module:: {{ obj.name }}

Welcome to the API Reference of PRIDE. This manual details functions, modules, and objects included in PRIDE, describing what they are and what they do.

{% block submodules %}
         {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
         {% set visible_submodules = obj.submodules|selectattr("display")|list %}
         {% set visible_submodules = (visible_subpackages + visible_submodules)|sort %}
         {% if visible_submodules %}
Submodules
----------

.. toctree::
   :hidden:
   :maxdepth: 1

            {% for submodule in visible_submodules %}
   {{ submodule.name|replace("pride.", "")|capitalize() }} <{{ submodule.include_path }}>
            {% endfor %}


.. autoapisummary::

            {% for submodule in visible_submodules %}
   	{{ submodule.id }}
            {% endfor %}


         {% endif %}
      {% endblock %}

   {% else %}
.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
   .. autoapi-nested-parse::

      {{ obj.docstring|indent(6) }}

      {% endif %}
      {% for obj_item in visible_children %}
   {{ obj_item.render()|indent(3) }}
      {% endfor %}
   {% endif %}
{% endif %}
